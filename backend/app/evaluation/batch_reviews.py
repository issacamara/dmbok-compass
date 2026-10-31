"""Immutable batch-evidence review snapshots and deterministic sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from typing import Any, Protocol

from google.api_core.exceptions import AlreadyExists

from app.contracts import (
    BatchAttestationRequest,
    DurableBatchAttestation,
    DurableExceptionDecision,
    DurableExceptionRecord,
    DurableRepresentativeSampleEntry,
    DurableReviewSummary,
    EvaluationDataset,
    EvaluationRun,
    RunBoundSupportEvidence,
)

DECISION_VERSION = "batch-review-v1"


class BatchReviewError(ValueError):
    """Base error for batch-review validation and persistence."""


class NoEligibleCandidatesError(BatchReviewError):
    """Raised when a run has no exact, verified support candidate."""


class BatchReviewConflictError(BatchReviewError):
    """Raised when immutable review or attestation identity is reused."""


class BatchReviewStaleError(BatchReviewError):
    """Raised when a review command does not match its immutable snapshot."""


@dataclass(frozen=True)
class BatchReview:
    """The immutable review snapshot plus its write-once child records."""

    summary: DurableReviewSummary
    samples: tuple[DurableRepresentativeSampleEntry, ...]
    exceptions: tuple[DurableExceptionRecord, ...]

    @property
    def key(self) -> str:
        return review_key(
            self.summary.generation_id,
            self.summary.run_id,
            self.summary.candidate_snapshot_sha256,
            self.summary.corpus_version_id,
        )


class BatchReviewRepository(Protocol):
    def create_or_get(self, review: BatchReview) -> BatchReview: ...

    def get(self, *, generation_id: str, run_id: str) -> BatchReview | None: ...

    def attest(
        self,
        review: BatchReview,
        request: BatchAttestationRequest,
        *,
        actor: str,
        decided_at: datetime,
    ) -> DurableBatchAttestation: ...


def review_key(
    generation_id: str,
    run_id: str,
    candidate_snapshot_sha256: str,
    corpus_version_id: str,
) -> str:
    """Return a stable, Firestore-safe identity for one immutable snapshot."""

    raw = f"{generation_id}|{run_id}|{candidate_snapshot_sha256}|{corpus_version_id}"
    return sha256(raw.encode("utf-8")).hexdigest()


def create_batch_review(
    run: EvaluationRun,
    dataset: EvaluationDataset,
    evidence: tuple[RunBoundSupportEvidence, ...] | list[RunBoundSupportEvidence],
    *,
    decision_version: str = DECISION_VERSION,
) -> BatchReview:
    """Build a content-safe review from one complete run-bound evidence set."""

    if run.candidate_snapshot is None:
        raise BatchReviewStaleError("batch review requires an immutable candidate snapshot")
    snapshot_digest = _candidate_snapshot_digest(run)
    if not run.selected_item_ids:
        raise BatchReviewError("batch review requires selected run items")
    if run.dataset_version_id != dataset.dataset_version_id:
        raise BatchReviewStaleError("dataset is not bound to this evaluation run")
    if len(set(run.selected_item_ids)) != len(run.selected_item_ids):
        raise BatchReviewError("run contains duplicate selected item IDs")

    by_item = {item.item_id: item for item in dataset.items}
    if len(by_item) != len(dataset.items):
        raise BatchReviewError("dataset contains duplicate item IDs")
    by_evidence: dict[str, RunBoundSupportEvidence] = {}
    for item_evidence in evidence:
        if item_evidence.item_id in by_evidence:
            raise BatchReviewError("evidence contains duplicate item IDs")
        _validate_evidence_binding(item_evidence, run, snapshot_digest)
        by_evidence[item_evidence.item_id] = item_evidence

    missing = [item_id for item_id in run.selected_item_ids if item_id not in by_evidence]
    if missing:
        raise BatchReviewError("batch review evidence is incomplete")
    extra = set(by_evidence) - set(run.selected_item_ids)
    if extra:
        raise BatchReviewError("evidence contains items outside the evaluation run")

    eligible: list[_Candidate] = []
    exceptions: list[DurableExceptionRecord] = []
    for item_id in run.selected_item_ids:
        item = by_item.get(item_id)
        item_evidence = by_evidence[item_id]
        if item is None:
            raise BatchReviewError("run item is not present in its dataset")
        if item_evidence.match_status == "eligible":
            eligible.append(_Candidate(item_id, item.category, "exact", item_evidence))
        else:
            exceptions.append(
                _exception_record(
                    run,
                    item_id,
                    item_evidence,
                    decision_version=decision_version,
                )
            )

    if not eligible:
        raise NoEligibleCandidatesError("batch review has no eligible candidate mappings")

    sampled = _select_sample(eligible)
    summary = DurableReviewSummary(
        generation_id=dataset.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=snapshot_digest,
        corpus_version_id=run.corpus_version_id,
        status="exception" if exceptions else "eligible",
        eligible_count=len(eligible),
        exception_count=len(exceptions),
        sampled_item_ids=[candidate.item_id for candidate in sampled],
        decision_version=decision_version,
    )
    samples = tuple(
        _sample_entry(run, candidate, decision_version=decision_version)
        for candidate in sampled
    )
    return BatchReview(summary=summary, samples=samples, exceptions=tuple(exceptions))


def _candidate_snapshot_digest(run: EvaluationRun) -> str:
    assert run.candidate_snapshot is not None
    return sha256(run.candidate_snapshot.model_dump_json().encode("utf-8")).hexdigest()


def _validate_evidence_binding(
    evidence: RunBoundSupportEvidence,
    run: EvaluationRun,
    snapshot_digest: str,
) -> None:
    if (
        evidence.run_id != run.run_id
        or evidence.corpus_version_id != run.corpus_version_id
        or evidence.candidate_snapshot_sha256 != snapshot_digest
    ):
        raise BatchReviewStaleError("support evidence is not bound to this evaluation snapshot")
    if evidence.match_status == "eligible":
        if evidence.selected_chunk is None or len(evidence.verified_chunks) != 1:
            raise BatchReviewError("eligible support must contain one verified selected chunk")
        if evidence.selected_chunk.content_hash != evidence.verified_chunks[0].content_hash:
            raise BatchReviewError("eligible support content hashes do not match")


@dataclass(frozen=True)
class _Candidate:
    item_id: str
    category: str
    confidence: str
    evidence: RunBoundSupportEvidence


def _select_sample(candidates: list[_Candidate]) -> tuple[_Candidate, ...]:
    sample_size = min(len(candidates), min(10, max(5, math.ceil(len(candidates) * 0.2))))
    strata: dict[tuple[str, str], list[_Candidate]] = {}
    for candidate in sorted(candidates, key=lambda value: (value.category, value.confidence, value.item_id)):
        strata.setdefault((candidate.category, candidate.confidence), []).append(candidate)

    selected: list[_Candidate] = []
    # Round-robin strata makes category/confidence coverage deterministic while
    # keeping the remaining slots stable by item ID.
    groups = [strata[key] for key in sorted(strata)]
    while groups and len(selected) < sample_size:
        next_groups: list[list[_Candidate]] = []
        for group in groups:
            if group and len(selected) < sample_size:
                selected.append(group.pop(0))
            if group:
                next_groups.append(group)
        groups = next_groups
    return tuple(selected)


def _identity(run: EvaluationRun, item_id: str) -> str:
    return sha256(f"{run.run_id}|{item_id}|{_candidate_snapshot_digest(run)}".encode()).hexdigest()


def _sample_entry(run: EvaluationRun, candidate: _Candidate, *, decision_version: str) -> DurableRepresentativeSampleEntry:
    evidence = candidate.evidence
    return DurableRepresentativeSampleEntry(
        generation_id=run.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=_candidate_snapshot_digest(run),
        corpus_version_id=run.corpus_version_id,
        sample_identity=_identity(run, candidate.item_id),
        item_id=candidate.item_id,
        candidate_chunk_ids=list(evidence.candidate_chunk_ids),
        content_hashes=[chunk.content_hash for chunk in evidence.verified_chunks],
        provenance_digests=[chunk.source_provenance_sha256 for chunk in evidence.verified_chunks],
        status="eligible",
        decision_version=decision_version,
    )


def _exception_record(
    run: EvaluationRun,
    item_id: str,
    evidence: RunBoundSupportEvidence,
    *,
    decision_version: str,
) -> DurableExceptionRecord:
    return DurableExceptionRecord(
        generation_id=run.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=_candidate_snapshot_digest(run),
        corpus_version_id=run.corpus_version_id,
        sample_identity=_identity(run, item_id),
        item_id=item_id,
        candidate_chunk_ids=list(evidence.candidate_chunk_ids),
        content_hashes=[chunk.content_hash for chunk in evidence.verified_chunks],
        provenance_digests=[chunk.source_provenance_sha256 for chunk in evidence.verified_chunks],
        status="inapplicable" if evidence.match_status == "inapplicable" else "exception",
        reason=evidence.reason or evidence.match_status,
        decision_version=decision_version,
    )


def _decision(
    review: BatchReview,
    request: BatchAttestationRequest,
    *,
    actor: str,
    decided_at: datetime,
) -> DurableBatchAttestation:
    if not actor:
        raise BatchReviewError("authenticated actor is required")
    if review.summary.status not in {"eligible", "exception"}:
        raise BatchReviewConflictError("batch review is already decided")
    expected_ids = {record.item_id for record in review.exceptions}
    actual_ids = {decision.item_id for decision in request.exception_decisions}
    if actual_ids != expected_ids:
        raise BatchReviewConflictError("all and only current exceptions require explicit decisions")
    if any(decision.decision == "attested" for decision in request.exception_decisions):
        raise BatchReviewConflictError("an exception cannot become approved support without a selected chunk")
    return DurableBatchAttestation(
        generation_id=review.summary.generation_id,
        run_id=review.summary.run_id,
        candidate_snapshot_sha256=review.summary.candidate_snapshot_sha256,
        corpus_version_id=review.summary.corpus_version_id,
        sampled_item_ids=list(review.summary.sampled_item_ids),
        decision=request.decision,
        actor=actor,
        decided_at=decided_at.astimezone(UTC),
        decision_version=review.summary.decision_version,
        exception_decisions=[
            DurableExceptionDecision(
                generation_id=review.summary.generation_id,
                run_id=review.summary.run_id,
                candidate_snapshot_sha256=review.summary.candidate_snapshot_sha256,
                corpus_version_id=review.summary.corpus_version_id,
                sample_identity=next(
                    record.sample_identity for record in review.exceptions if record.item_id == decision.item_id
                ),
                item_id=decision.item_id,
                decision=decision.decision,
                actor=actor,
                decided_at=decided_at.astimezone(UTC),
                decision_version=review.summary.decision_version,
                reason=decision.reason,
            )
            for decision in request.exception_decisions
        ],
    )


class InMemoryBatchReviewRepository:
    """Thread-safe restart seam used by local and contract tests."""

    def __init__(self) -> None:
        self._reviews: dict[str, BatchReview] = {}
        self._attestations: dict[str, DurableBatchAttestation] = {}
        self._lock = RLock()

    def create_or_get(self, review: BatchReview) -> BatchReview:
        with self._lock:
            existing = self._reviews.get(review.key)
            if existing is not None:
                if existing != review:
                    raise BatchReviewConflictError("review snapshot is already bound to different evidence")
                return existing
            self._reviews[review.key] = review
            return review

    def get(self, *, generation_id: str, run_id: str) -> BatchReview | None:
        with self._lock:
            return next(
                (review for review in self._reviews.values()
                 if review.summary.generation_id == generation_id and review.summary.run_id == run_id),
                None,
            )

    def attest(
        self,
        review: BatchReview,
        request: BatchAttestationRequest,
        *,
        actor: str,
        decided_at: datetime,
    ) -> DurableBatchAttestation:
        with self._lock:
            stored = self._reviews.get(review.key)
            if stored is None or stored != review:
                raise BatchReviewStaleError("batch review is not the current immutable snapshot")
            if review.key in self._attestations:
                raise BatchReviewConflictError("batch review has already been decided")
            result = _decision(review, request, actor=actor, decided_at=decided_at)
            self._attestations[review.key] = result
            return result

    def attestation(self, review: BatchReview) -> DurableBatchAttestation | None:
        return self._attestations.get(review.key)


class FirestoreBatchReviewRepository:
    """Firestore-backed immutable review records and append-only decisions."""

    summary_collection_id = "evaluation_batch_reviews"
    sample_collection_id = "evaluation_batch_review_samples"
    exception_collection_id = "evaluation_batch_review_exceptions"
    attestation_collection_id = "evaluation_batch_attestations"

    def __init__(self, client: Any) -> None:
        self.summaries = client.collection(self.summary_collection_id)
        self.samples = client.collection(self.sample_collection_id)
        self.exceptions = client.collection(self.exception_collection_id)
        self.attestations = client.collection(self.attestation_collection_id)

    def create_or_get(self, review: BatchReview) -> BatchReview:
        reference = self.summaries.document(review.key)
        payload = review.summary.model_dump(mode="json")
        try:
            reference.create(payload)
            for entry in review.samples:
                self.samples.document(f"{review.key}:{entry.item_id}").create(entry.model_dump(mode="json"))
            for record in review.exceptions:
                self.exceptions.document(f"{review.key}:{record.item_id}").create(record.model_dump(mode="json"))
            return review
        except AlreadyExists:
            existing = self.get(generation_id=review.summary.generation_id, run_id=review.summary.run_id)
            if existing is None or existing.summary != review.summary or existing.samples != review.samples or existing.exceptions != review.exceptions:
                raise BatchReviewConflictError("review snapshot is already bound to different evidence")
            return existing

    def get(self, *, generation_id: str, run_id: str) -> BatchReview | None:
        # The document ID is derived from the candidate/corpus binding, so use a
        # bounded collection scan only for this server-owned read seam.
        for snapshot in self.summaries.stream():
            if not snapshot.exists:
                continue
            summary = DurableReviewSummary.model_validate(snapshot.to_dict())
            if summary.generation_id != generation_id or summary.run_id != run_id:
                continue
            sample_payloads = [
                snapshot.to_dict()
                for snapshot in self.samples.stream()
                if snapshot.exists
                and snapshot.to_dict().get("run_id") == run_id
                and snapshot.to_dict().get("candidate_snapshot_sha256") == summary.candidate_snapshot_sha256
            ]
            exception_payloads = [
                snapshot.to_dict()
                for snapshot in self.exceptions.stream()
                if snapshot.exists
                and snapshot.to_dict().get("run_id") == run_id
                and snapshot.to_dict().get("candidate_snapshot_sha256") == summary.candidate_snapshot_sha256
            ]
            samples = tuple(
                DurableRepresentativeSampleEntry.model_validate(payload)
                for payload in sorted(sample_payloads, key=lambda value: value["item_id"])
            )
            exceptions = tuple(
                DurableExceptionRecord.model_validate(payload)
                for payload in sorted(exception_payloads, key=lambda value: value["item_id"])
            )
            return BatchReview(summary=summary, samples=samples, exceptions=exceptions)
        return None

    def attest(
        self,
        review: BatchReview,
        request: BatchAttestationRequest,
        *,
        actor: str,
        decided_at: datetime,
    ) -> DurableBatchAttestation:
        current = self.get(generation_id=review.summary.generation_id, run_id=review.summary.run_id)
        if current is None or current != review:
            raise BatchReviewStaleError("batch review is not the current immutable snapshot")
        result = _decision(review, request, actor=actor, decided_at=decided_at)
        reference = self.attestations.document(review.key)
        try:
            reference.create(result.model_dump(mode="json"))
        except AlreadyExists as exc:
            raise BatchReviewConflictError("batch review has already been decided") from exc
        return result
