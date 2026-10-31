"""Run-bound automatic support preparation.

Support preparation is deliberately separate from scoring.  It turns authored
passages into durable, candidate-pinned evidence without approving any
answer, claim, citation, or rubric assessment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import re
from threading import RLock
from typing import Any, Protocol

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from app.contracts import (
    DocumentChunk,
    EvaluationDataset,
    EvaluationPreparationComplete,
    EvaluationRun,
    RunBoundSupportEvidence,
    VerifiedChunkChoice,
)


class SupportPreparationError(ValueError):
    """Raised when a run cannot be safely bound to support evidence."""


@dataclass(frozen=True)
class StoredSupportChunk:
    """A chunk plus the server-owned metadata used to verify it."""

    chunk: DocumentChunk | None
    chunk_id: str
    source_provenance_sha256: str
    valid: bool = True
    invalid_reason: str | None = None


class RunPinnedChunkSource(Protocol):
    def chunks(self, *, corpus_version_id: str) -> tuple[StoredSupportChunk, ...]: ...

    def source_provenance_sha256(self, *, corpus_version_id: str) -> str: ...


class SupportEvidenceStore(Protocol):
    def get_evidence(self, *, run_id: str, item_id: str) -> RunBoundSupportEvidence | None: ...

    def create_evidence(self, evidence: RunBoundSupportEvidence) -> RunBoundSupportEvidence: ...

    def get_marker(self, *, run_id: str) -> EvaluationPreparationComplete | None: ...

    def create_marker(self, marker: EvaluationPreparationComplete) -> EvaluationPreparationComplete: ...

    def review_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        selected_chunk: VerifiedChunkChoice | None,
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence: ...


@dataclass(frozen=True)
class SupportPreparationResult:
    evidence: tuple[RunBoundSupportEvidence, ...]
    marker: EvaluationPreparationComplete


class InMemorySupportEvidenceStore:
    """Restart-safe store used by unit tests and local jobs."""

    def __init__(self) -> None:
        self._evidence: dict[tuple[str, str], RunBoundSupportEvidence] = {}
        self._markers: dict[str, EvaluationPreparationComplete] = {}
        self._lock = RLock()

    def get_evidence(self, *, run_id: str, item_id: str) -> RunBoundSupportEvidence | None:
        with self._lock:
            return deepcopy(self._evidence.get((run_id, item_id)))

    def create_evidence(self, evidence: RunBoundSupportEvidence) -> RunBoundSupportEvidence:
        key = (evidence.run_id, evidence.item_id)
        with self._lock:
            existing = self._evidence.get(key)
            if existing is not None:
                if existing.model_dump() != evidence.model_dump():
                    raise SupportPreparationError("support evidence is already bound to different data")
                return deepcopy(existing)
            self._evidence[key] = evidence.model_copy(deep=True)
            return deepcopy(evidence)

    def get_marker(self, *, run_id: str) -> EvaluationPreparationComplete | None:
        with self._lock:
            return deepcopy(self._markers.get(run_id))

    def create_marker(self, marker: EvaluationPreparationComplete) -> EvaluationPreparationComplete:
        with self._lock:
            existing = self._markers.get(marker.run_id)
            if existing is not None:
                if existing.model_dump() != marker.model_dump():
                    raise SupportPreparationError("preparation marker is already bound to different data")
                return deepcopy(existing)
            self._markers[marker.run_id] = marker.model_copy(deep=True)
            return deepcopy(marker)

    def review_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        selected_chunk: VerifiedChunkChoice | None,
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence:
        with self._lock:
            current = self._evidence.get((evidence.run_id, evidence.item_id))
            if current is None:
                raise SupportPreparationError("support evidence was not prepared")
            return _review_evidence(
                current,
                selected_chunk=selected_chunk,
                reviewer_actor=reviewer_actor,
                reviewed_at=reviewed_at,
                save=lambda updated: self._evidence.__setitem__(
                    (updated.run_id, updated.item_id), updated.model_copy(deep=True)
                ),
            )


class FirestoreSupportEvidenceStore:
    """Durable per-item evidence and completion markers."""

    evidence_collection_id = "evaluation_support_evidence"
    marker_collection_id = "evaluation_preparation"

    def __init__(self, client: Any) -> None:
        self.client = client
        self.evidence = client.collection(self.evidence_collection_id)
        self.markers = client.collection(self.marker_collection_id)

    @staticmethod
    def _evidence_id(run_id: str, item_id: str) -> str:
        return f"{run_id}:{item_id}"

    def get_evidence(self, *, run_id: str, item_id: str) -> RunBoundSupportEvidence | None:
        snapshot = self.evidence.document(self._evidence_id(run_id, item_id)).get()
        return RunBoundSupportEvidence.model_validate(snapshot.to_dict()) if snapshot.exists else None

    def create_evidence(self, evidence: RunBoundSupportEvidence) -> RunBoundSupportEvidence:
        reference = self.evidence.document(self._evidence_id(evidence.run_id, evidence.item_id))
        payload = evidence.model_dump(mode="json")
        try:
            reference.create(payload)
        except AlreadyExists:
            existing = self.get_evidence(run_id=evidence.run_id, item_id=evidence.item_id)
            if existing is None or existing.model_dump() != evidence.model_dump():
                raise SupportPreparationError("support evidence is already bound to different data") from None
            return existing
        return evidence.model_copy(deep=True)

    def get_marker(self, *, run_id: str) -> EvaluationPreparationComplete | None:
        snapshot = self.markers.document(run_id).get()
        return EvaluationPreparationComplete.model_validate(snapshot.to_dict()) if snapshot.exists else None

    def create_marker(self, marker: EvaluationPreparationComplete) -> EvaluationPreparationComplete:
        reference = self.markers.document(marker.run_id)
        payload = marker.model_dump(mode="json")
        try:
            reference.create(payload)
        except AlreadyExists:
            existing = self.get_marker(run_id=marker.run_id)
            if existing is None or existing.model_dump() != marker.model_dump():
                raise SupportPreparationError("preparation marker is already bound to different data") from None
            return existing
        return marker.model_copy(deep=True)

    def review_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        selected_chunk: VerifiedChunkChoice | None,
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence:
        reference = self.evidence.document(self._evidence_id(evidence.run_id, evidence.item_id))

        @firestore.transactional
        def apply(transaction: Any) -> RunBoundSupportEvidence:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise SupportPreparationError("support evidence was not prepared")
            current = RunBoundSupportEvidence.model_validate(snapshot.to_dict())
            return _review_evidence(
                current,
                selected_chunk=selected_chunk,
                reviewer_actor=reviewer_actor,
                reviewed_at=reviewed_at,
                save=lambda updated: transaction.set(reference, updated.model_dump(mode="json")),
            )

        return apply(self.client.transaction())


class FirestoreRunPinnedChunkSource:
    """Read and verify only chunks in the run's pinned corpus version."""

    def __init__(
        self,
        client: Any,
        *,
        vector_collection: str,
        provenance_by_corpus: dict[str, str],
        ingestion_fingerprint: str,
        configuration_fingerprint: str,
    ) -> None:
        self.client = client
        self.vector_collection = vector_collection
        self.provenance_by_corpus = provenance_by_corpus
        self.ingestion_fingerprint = ingestion_fingerprint
        self.configuration_fingerprint = configuration_fingerprint

    def source_provenance_sha256(self, *, corpus_version_id: str) -> str:
        try:
            value = self.provenance_by_corpus[corpus_version_id]
        except KeyError as exc:
            raise SupportPreparationError("run corpus provenance is unavailable") from exc
        return value

    def chunk(self, *, corpus_version_id: str, chunk_id: str) -> StoredSupportChunk | None:
        return next(
            (record for record in self.chunks(corpus_version_id=corpus_version_id) if record.chunk_id == chunk_id),
            None,
        )

    def chunks(self, *, corpus_version_id: str) -> tuple[StoredSupportChunk, ...]:
        provenance = self.source_provenance_sha256(corpus_version_id=corpus_version_id)
        records: list[StoredSupportChunk] = []
        snapshots = self.client.collection(self.vector_collection).where(
            filter=FieldFilter("metadata.corpus_version_id", "==", corpus_version_id)
        ).stream()
        for snapshot in snapshots:
            payload = snapshot.to_dict()
            metadata = payload.get("metadata", {})
            chunk_id = str(metadata.get("chunk_id") or getattr(snapshot, "id", ""))
            try:
                if not isinstance(metadata, dict) or metadata.get("corpus_version_id") != corpus_version_id:
                    raise ValueError("chunk corpus metadata is invalid")
                if (
                    metadata.get("corpus_manifest_sha256") != self.ingestion_fingerprint
                    or metadata.get("derivation_sha256") != self.configuration_fingerprint
                ):
                    raise ValueError("chunk source metadata is incomplete")
                text = str(payload["text"])
                content_hash = str(metadata["content_hash"])
                if sha256(text.encode("utf-8")).hexdigest() != content_hash:
                    raise ValueError("chunk content hash is invalid")
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    corpus_version_id=str(metadata["corpus_version_id"]),
                    page=int(metadata["page"]),
                    section=str(metadata["section"]),
                    chunk_ordinal=int(metadata["chunk_ordinal"]),
                    content_hash=content_hash,
                    text=text,
                    embedding=list(payload["embedding"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                records.append(StoredSupportChunk(None, chunk_id, provenance, False, str(exc)))
            else:
                records.append(StoredSupportChunk(chunk, chunk_id, provenance))
        return tuple(sorted(records, key=lambda record: record.chunk_id))


def prepare_support(
    run: EvaluationRun,
    dataset: EvaluationDataset,
    chunks: RunPinnedChunkSource,
    store: SupportEvidenceStore,
) -> SupportPreparationResult:
    """Prepare all selected items, writing no completion marker on failure."""

    if run.candidate_snapshot is None:
        raise SupportPreparationError("immutable candidate snapshot is unavailable")
    if run.candidate_snapshot.corpus_binding.corpus_version_id != run.corpus_version_id:
        raise SupportPreparationError("candidate snapshot is not bound to the run corpus")
    marker = store.get_marker(run_id=run.run_id)
    if marker is not None:
        existing = tuple(
            store.get_evidence(run_id=run.run_id, item_id=item_id)
            for item_id in run.selected_item_ids
        )
        if all(item is not None for item in existing):
            return SupportPreparationResult(existing, marker)  # type: ignore[arg-type]
        raise SupportPreparationError("preparation marker exists without complete item evidence")

    source_provenance = chunks.source_provenance_sha256(corpus_version_id=run.corpus_version_id)
    candidate_chunks = tuple(
        _verify_record(record, run.corpus_version_id, source_provenance)
        for record in chunks.chunks(corpus_version_id=run.corpus_version_id)
    )
    annotations = {annotation.item_id: annotation for annotation in dataset.gold_annotations}
    prepared: list[RunBoundSupportEvidence] = []
    for item_id in run.selected_item_ids:
        annotation = annotations.get(item_id)
        evidence = _prepare_item(run, item_id, annotation, candidate_chunks, source_provenance)
        prepared.append(store.create_evidence(evidence))

    eligible = sum(item.match_status == "eligible" for item in prepared)
    marker = EvaluationPreparationComplete(
        run_id=run.run_id,
        candidate_snapshot_sha256=_snapshot_digest(run),
        corpus_version_id=run.corpus_version_id,
        item_count=len(prepared),
        prepared_item_count=len(prepared),
        approved_item_count=0,
        eligible_item_count=eligible,
        unresolved_item_count=len(prepared) - eligible,
        completed_at=datetime.now(timezone.utc),
    )
    return SupportPreparationResult(tuple(prepared), store.create_marker(marker))


def candidate_snapshot_sha256(run: EvaluationRun) -> str:
    """Return the digest used to bind support evidence to one evaluation run."""

    return _snapshot_digest(run)


def approved_support_chunk_ids(
    evidence: RunBoundSupportEvidence | None,
    *,
    run: EvaluationRun,
) -> frozenset[str]:
    """Return approved support IDs only when evidence is bound to this run."""

    if evidence is None:
        return frozenset()
    if (
        evidence.run_id != run.run_id
        or evidence.corpus_version_id != run.corpus_version_id
        or evidence.candidate_snapshot_sha256 != candidate_snapshot_sha256(run)
        or evidence.match_status != "review_approved"
    ):
        return frozenset()
    if evidence.selected_chunk is None:
        return frozenset()
    if evidence.selected_chunk.corpus_version_id != run.corpus_version_id:
        return frozenset()
    return frozenset({evidence.selected_chunk.chunk_id})


def review_support(
    evidence: RunBoundSupportEvidence,
    *,
    candidate: StoredSupportChunk | None,
    reviewer_actor: str,
    reviewed_at: datetime,
    store: SupportEvidenceStore,
) -> RunBoundSupportEvidence:
    """Apply one administrator decision after the candidate was re-read."""

    selected = None
    if candidate is not None:
        if not candidate.valid or candidate.chunk is None:
            raise SupportPreparationError("selected chunk failed server verification")
        selected = _choice(candidate, evidence.source_provenance_sha256)
    return store.review_evidence(
        evidence,
        selected_chunk=selected,
        reviewer_actor=reviewer_actor,
        reviewed_at=reviewed_at,
    )


def _review_evidence(
    evidence: RunBoundSupportEvidence,
    *,
    selected_chunk: VerifiedChunkChoice | None,
    reviewer_actor: str,
    reviewed_at: datetime,
    save: Any,
) -> RunBoundSupportEvidence:
    if evidence.match_status == "eligible":
        raise SupportPreparationError("eligible support requires batch attestation before approval")
    if evidence.match_status in {"review_approved", "review_rejected"}:
        same = evidence.selected_chunk == selected_chunk
        if evidence.match_status == "review_rejected":
            same = selected_chunk is None
        if not same:
            raise SupportPreparationError("support review is already bound to a different decision")
        return evidence
    if selected_chunk is not None and selected_chunk.chunk_id not in evidence.candidate_chunk_ids:
        raise SupportPreparationError("selected chunk is not a server-listed candidate")
    updated = evidence.model_copy(
        update={
            "match_status": "review_approved" if selected_chunk is not None else "review_rejected",
            "verified_chunks": [selected_chunk] if selected_chunk is not None else [],
            "selected_chunk": selected_chunk,
            "reviewer_actor": reviewer_actor,
            "reviewed_at": reviewed_at,
        }
    )
    save(updated)
    return updated


def _prepare_item(
    run: EvaluationRun,
    item_id: str,
    annotation: Any,
    records: tuple[StoredSupportChunk, ...],
    source_provenance: str,
) -> RunBoundSupportEvidence:
    snapshot_digest = _snapshot_digest(run)
    if annotation is None:
        return _unresolved(run, item_id, snapshot_digest, source_provenance, "missing_annotation")
    expected_refusal = bool(
        annotation.semantic_assessment is not None
        and annotation.semantic_assessment.expected_refusal is True
    )
    if expected_refusal:
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance,
            "explicit_unanswerable", expected_refusal=True,
        )
    passage = _normalise(annotation.supporting_passage or "")
    if not passage:
        return _unresolved(run, item_id, snapshot_digest, source_provenance, "missing_supporting_passage")
    invalid = [record for record in records if not record.valid]
    placeholder_ids = {
        record.chunk_id for record in records if _is_placeholder_chunk_id(record.chunk_id)
    }
    referenced_ids = set(annotation.relevant_chunk_ids)
    if referenced_ids & placeholder_ids:
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance, "placeholder_chunk_id",
            sorted(referenced_ids & placeholder_ids),
        )
    record_ids = {record.chunk_id for record in records}
    if referenced_ids and not referenced_ids.issubset(record_ids):
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance, "missing_chunk",
            sorted(referenced_ids - record_ids),
        )
    invalid_matches = [
        record for record in invalid
        if record.chunk is not None and passage in _normalise(record.chunk.text)
    ]
    if invalid_matches:
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance, "invalid_chunk_metadata",
            [record.chunk_id for record in invalid_matches],
        )
    matches = [
        record for record in records
        if record.valid and record.chunk is not None and passage in _normalise(record.chunk.text)
    ]
    if invalid and any(record.chunk_id in annotation.relevant_chunk_ids for record in invalid):
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance,
            "invalid_chunk_metadata", [record.chunk_id for record in invalid],
        )
    if len(matches) == 1:
        record = matches[0]
        assert record.chunk is not None
        choice = _choice(record, source_provenance)
        return RunBoundSupportEvidence(
            run_id=run.run_id,
            item_id=item_id,
            candidate_snapshot_sha256=snapshot_digest,
            corpus_version_id=run.corpus_version_id,
            source_provenance_sha256=source_provenance,
            match_status="eligible",
            candidate_chunk_ids=[record.chunk.chunk_id],
            verified_chunks=[choice],
            selected_chunk=choice,
        )
    if len(matches) > 1:
        return _unresolved(
            run, item_id, snapshot_digest, source_provenance, "ambiguous_support",
            [record.chunk_id for record in matches],
        )
    boundary = _boundary_candidates(passage, records)
    return _unresolved(
        run, item_id, snapshot_digest, source_provenance,
        "support_spans_chunk_boundary" if boundary else "missing_supporting_passage",
        boundary,
    )


def _unresolved(
    run: EvaluationRun,
    item_id: str,
    snapshot_digest: str,
    provenance: str,
    reason: str,
    candidates: list[str] | None = None,
    *,
    expected_refusal: bool = False,
) -> RunBoundSupportEvidence:
    status = (
        "ambiguous" if reason == "ambiguous_support"
        else "inapplicable" if expected_refusal
        else "invalid" if reason in {"invalid_chunk_metadata", "placeholder_chunk_id", "invalid_provenance"}
        else "missing"
    )
    return RunBoundSupportEvidence(
        run_id=run.run_id,
        item_id=item_id,
        candidate_snapshot_sha256=snapshot_digest,
        corpus_version_id=run.corpus_version_id,
        source_provenance_sha256=provenance,
        match_status=status,
        candidate_chunk_ids=list(dict.fromkeys(candidates or []))[:5],
        reason=reason,
        expected_refusal=expected_refusal,
    )


def _choice(record: StoredSupportChunk, provenance: str) -> VerifiedChunkChoice:
    if record.chunk is None or not record.valid:
        raise SupportPreparationError("cannot verify an invalid chunk")
    return VerifiedChunkChoice(
        chunk_id=record.chunk.chunk_id,
        corpus_version_id=record.chunk.corpus_version_id,
        content_hash=record.chunk.content_hash,
        page=record.chunk.page,
        section=record.chunk.section,
        source_provenance_sha256=provenance,
    )


def _verify_record(record: StoredSupportChunk, corpus_version_id: str, provenance: str) -> StoredSupportChunk:
    """Apply integrity checks even when the source is an in-memory adapter."""

    if not record.valid or record.chunk is None:
        return record
    chunk = record.chunk
    reason: str | None = None
    if _is_placeholder_chunk_id(record.chunk_id) or record.chunk_id != chunk.chunk_id:
        reason = "chunk identity is invalid"
    elif chunk.corpus_version_id != corpus_version_id:
        reason = "chunk corpus metadata is invalid"
    elif record.source_provenance_sha256 != provenance:
        reason = "invalid provenance"
    elif sha256(chunk.text.encode("utf-8")).hexdigest() != chunk.content_hash:
        reason = "chunk content hash is invalid"
    if reason is not None:
        return StoredSupportChunk(chunk, record.chunk_id, record.source_provenance_sha256, False, reason)
    return record


def _is_placeholder_chunk_id(chunk_id: str) -> bool:
    """Reject IDs created by imports instead of the immutable corpus."""

    return chunk_id.startswith("imported-passage-") or chunk_id.startswith("placeholder-")


def _boundary_candidates(passage: str, records: tuple[StoredSupportChunk, ...]) -> list[str]:
    valid = [record for record in records if record.valid and record.chunk is not None]
    for left, right in zip(valid, valid[1:]):
        assert left.chunk is not None and right.chunk is not None
        joined = _normalise(left.chunk.text + " " + right.chunk.text)
        if (
            passage in joined
            and passage not in _normalise(left.chunk.text)
            and passage not in _normalise(right.chunk.text)
        ):
            return [left.chunk.chunk_id, right.chunk.chunk_id]
    return []


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _snapshot_digest(run: EvaluationRun) -> str:
    if run.candidate_snapshot is None:
        raise SupportPreparationError("immutable candidate snapshot is unavailable")
    return sha256(run.candidate_snapshot.model_dump_json().encode("utf-8")).hexdigest()
