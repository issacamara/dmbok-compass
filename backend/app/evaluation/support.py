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
from typing import Any, Literal, Protocol

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from app.contracts import (
    ChildDocumentChunk,
    DocumentChunk,
    EvaluationDataset,
    EvaluationPreparationComplete,
    EvaluationRun,
    ParentChunk,
    RunBoundSupportEvidence,
    VerifiedChunkChoice,
)
from app.corpus.indexing import PARENT_COLLECTION
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations


class SupportPreparationError(ValueError):
    """Raised when a run cannot be safely bound to support evidence."""


@dataclass(frozen=True)
class StoredSupportChunk:
    """A chunk plus the server-owned metadata used to verify it."""

    chunk: DocumentChunk | ChildDocumentChunk | None
    chunk_id: str
    source_provenance_sha256: str
    valid: bool = True
    invalid_reason: str | None = None
    parent: ParentChunk | None = None


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
        selected_chunks: tuple[VerifiedChunkChoice, ...],
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence: ...

    def decide_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        decision: Literal["approve", "reject"],
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
        selected_chunks: tuple[VerifiedChunkChoice, ...],
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence:
        with self._lock:
            current = self._evidence.get((evidence.run_id, evidence.item_id))
            if current is None:
                raise SupportPreparationError("support evidence was not prepared")
            return _review_evidence(
                current,
                selected_chunks=selected_chunks,
                reviewer_actor=reviewer_actor,
                reviewed_at=reviewed_at,
                save=lambda updated: self._evidence.__setitem__(
                    (updated.run_id, updated.item_id), updated.model_copy(deep=True)
                ),
            )

    def decide_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        decision: Literal["approve", "reject"],
        reviewer_actor: str,
        reviewed_at: datetime,
    ) -> RunBoundSupportEvidence:
        with self._lock:
            current = self._evidence.get((evidence.run_id, evidence.item_id))
            if current is None:
                raise SupportPreparationError("support evidence was not prepared")
            return _decide_support(
                current,
                decision=decision,
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
        selected_chunks: tuple[VerifiedChunkChoice, ...],
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
                selected_chunks=selected_chunks,
                reviewer_actor=reviewer_actor,
                reviewed_at=reviewed_at,
                save=lambda updated: transaction.set(reference, updated.model_dump(mode="json")),
            )

        return apply(self.client.transaction())

    def decide_evidence(
        self,
        evidence: RunBoundSupportEvidence,
        *,
        decision: Literal["approve", "reject"],
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
            return _decide_support(
                current,
                decision=decision,
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
        self._hierarchy_errors: dict[str, str | None] = {}

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
        parents: dict[str, ParentChunk] = {}
        parent_snapshots = self.client.collection(PARENT_COLLECTION).where(
            filter=FieldFilter("corpus_version_id", "==", corpus_version_id)
        ).stream()
        hierarchy_error: str | None = None
        for snapshot in parent_snapshots:
            payload = snapshot.to_dict()
            metadata = payload.get("metadata")
            parent_id = str(payload.get("parent_id") or getattr(snapshot, "id", ""))
            try:
                if not isinstance(metadata, dict) or (
                    metadata.get("corpus_manifest_sha256") != self.ingestion_fingerprint
                    or metadata.get("derivation_sha256") != self.configuration_fingerprint
                ):
                    raise ValueError("parent source metadata is incomplete")
                parent = ParentChunk.model_validate({key: value for key, value in payload.items() if key != "metadata"})
                if parent.parent_id != parent_id or sha256(parent.text.encode("utf-8")).hexdigest() != parent.content_hash:
                    raise ValueError("parent identity or content hash is invalid")
                if parent.parent_id in parents:
                    raise ValueError("parent IDs are duplicated")
            except (TypeError, ValueError) as exc:
                hierarchy_error = str(exc)
                continue
            parents[parent.parent_id] = parent

        records: list[StoredSupportChunk] = []
        snapshots = self.client.collection(self.vector_collection).where(
            filter=FieldFilter("metadata.corpus_version_id", "==", corpus_version_id)
        ).stream()
        for snapshot in snapshots:
            payload = snapshot.to_dict()
            metadata = payload.get("metadata")
            chunk_id = str(metadata.get("chunk_id") or getattr(snapshot, "id", "")) if isinstance(metadata, dict) else str(getattr(snapshot, "id", ""))
            try:
                if not isinstance(metadata, dict) or metadata.get("corpus_version_id") != corpus_version_id:
                    raise ValueError("child corpus metadata is invalid")
                if (
                    metadata.get("corpus_manifest_sha256") != self.ingestion_fingerprint
                    or metadata.get("derivation_sha256") != self.configuration_fingerprint
                ):
                    raise ValueError("child source metadata is incomplete")
                if metadata.get("schema_version") != "child-document-chunk-v1":
                    raise ValueError("run-bound support requires hierarchical child records")
                fields = (
                    "schema_version", "chunk_id", "corpus_version_id", "parent_id", "section_path",
                    "page_start", "page_end", "source_spans", "child_ordinal", "content_hash", "token_count",
                )
                child = ChildDocumentChunk.model_validate(
                    {**{key: metadata[key] for key in fields}, "text": payload["text"]}
                )
                if child.chunk_id != chunk_id or sha256(child.text.encode("utf-8")).hexdigest() != child.content_hash:
                    raise ValueError("child identity or content hash is invalid")
                parent = parents.get(child.parent_id)
                if parent is None:
                    hierarchy_error = "child references a missing or invalid parent"
                    records.append(StoredSupportChunk(child, chunk_id, provenance, False, hierarchy_error))
                    continue
                records.append(StoredSupportChunk(child, chunk_id, provenance, parent=parent))
            except (KeyError, TypeError, ValueError) as exc:
                hierarchy_error = str(exc)
                records.append(StoredSupportChunk(None, chunk_id, provenance, False, str(exc)))
        children = tuple(
            record.chunk for record in records
            if record.valid and isinstance(record.chunk, ChildDocumentChunk)
        )
        if hierarchy_error is None:
            try:
                from app.corpus.versions import (
                    hierarchical_provenance_sha256,
                    validate_hierarchical_records,
                )

                validate_hierarchical_records(corpus_version_id, tuple(parents.values()), children)
                if hierarchical_provenance_sha256(tuple(parents.values()), children) != provenance:
                    hierarchy_error = "hierarchical source provenance differs from the pinned corpus"
            except (ValueError, TypeError) as exc:
                hierarchy_error = str(exc)
        if hierarchy_error:
            records = [
                StoredSupportChunk(
                    record.chunk,
                    record.chunk_id,
                    record.source_provenance_sha256,
                    False,
                    hierarchy_error,
                    record.parent,
                )
                for record in records
            ]
        self._hierarchy_errors[corpus_version_id] = hierarchy_error
        return tuple(sorted(records, key=lambda record: record.chunk_id))

    def hierarchy_error(self, *, corpus_version_id: str) -> str | None:
        return self._hierarchy_errors.get(corpus_version_id)


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
    hierarchy_error_method = getattr(chunks, "hierarchy_error", None)
    hierarchy_error = hierarchy_error_method(corpus_version_id=run.corpus_version_id) if callable(hierarchy_error_method) else None
    hierarchical = any(isinstance(record.chunk, ChildDocumentChunk) for record in candidate_chunks)
    if hierarchy_error:
        prepared = [
            store.create_evidence(
                _unresolved(
                    run, item_id, _snapshot_digest(run), source_provenance,
                    "invalid_parent_child_hierarchy",
                )
            )
            for item_id in run.selected_item_ids
        ]
    elif hierarchical:
        parents = tuple({record.parent.parent_id: record.parent for record in candidate_chunks if record.parent}.values())
        children = tuple(
            record.chunk for record in candidate_chunks
            if isinstance(record.chunk, ChildDocumentChunk) and record.valid
        )
        if len(children) != len(candidate_chunks):
            prepared = [
                store.create_evidence(
                    _unresolved(
                        run, item_id, _snapshot_digest(run), source_provenance,
                        "invalid_parent_child_hierarchy",
                    )
                )
                for item_id in run.selected_item_ids
            ]
        else:
            try:
                binding = CandidateGoldBinding(
                    corpus_version_id=run.corpus_version_id,
                    candidate_snapshot_sha256=_snapshot_digest(run),
                    source_provenance_sha256=source_provenance,
                )
                remapped = remap_gold_annotations(dataset, (*parents, *children), binding)
            except (ValueError, TypeError) as exc:
                raise SupportPreparationError("parent/child support mapping failed validation") from exc
            records_by_id = {record.chunk_id: record for record in candidate_chunks}
            for item_id in run.selected_item_ids:
                annotation = annotations.get(item_id)
                remap = next((entry for entry in remapped.remaps if entry.item_id == item_id), None)
                if annotation is None or remap is None:
                    evidence = _unresolved(
                        run, item_id, _snapshot_digest(run), source_provenance, "missing_annotation"
                    )
                elif (
                    annotation.semantic_assessment is not None
                    and annotation.semantic_assessment.expected_refusal is True
                ):
                    evidence = _unresolved(
                        run, item_id, _snapshot_digest(run), source_provenance,
                        "explicit_unanswerable", expected_refusal=True,
                    )
                elif remap.remap_status == "matched":
                    selected = tuple(verified_chunk_choice(records_by_id[chunk_id], source_provenance) for chunk_id in remap.relevant_chunk_ids)
                    evidence = RunBoundSupportEvidence(
                        run_id=run.run_id,
                        item_id=item_id,
                        candidate_snapshot_sha256=_snapshot_digest(run),
                        corpus_version_id=run.corpus_version_id,
                        source_provenance_sha256=source_provenance,
                        match_status="eligible",
                        candidate_chunk_ids=[choice.chunk_id for choice in selected],
                        verified_chunks=list(selected),
                        selected_chunk=selected[0] if len(selected) == 1 else None,
                        selected_chunks=list(selected),
                        parent_id=remap.parent_id,
                        parent_content_hash=remap.parent_content_hash,
                        parent_span_sha256=remap.parent_span_sha256,
                        binding_version="run-bound-support-v2",
                    )
                else:
                    reason = remap.review_reasons[0] if remap.review_reasons else remap.remap_status
                    evidence = _unresolved(
                        run, item_id, _snapshot_digest(run), source_provenance,
                        "ambiguous_support" if remap.remap_status == "ambiguous" else reason,
                        list(remap.candidate_chunk_ids),
                    )
                prepared.append(store.create_evidence(evidence))
    for item_id in run.selected_item_ids:
        if not hierarchical and not hierarchy_error:
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
    if evidence.expected_refusal and evidence.reviewer_actor and evidence.reviewed_at is not None:
        return frozenset()
    selected = evidence.selected_chunks or ([evidence.selected_chunk] if evidence.selected_chunk is not None else [])
    return frozenset(choice.chunk_id for choice in selected if choice.corpus_version_id == run.corpus_version_id)


def approved_refusal(evidence: RunBoundSupportEvidence | None, *, run: EvaluationRun) -> bool:
    return bool(
        evidence is not None
        and evidence.match_status == "review_approved"
        and evidence.expected_refusal
        and not evidence.candidate_chunk_ids
        and not evidence.verified_chunks
        and not evidence.selected_chunks
        and evidence.selected_chunk is None
        and evidence.run_id == run.run_id
        and evidence.corpus_version_id == run.corpus_version_id
        and evidence.candidate_snapshot_sha256 == candidate_snapshot_sha256(run)
    )


def review_support(
    evidence: RunBoundSupportEvidence,
    *,
    candidates: tuple[StoredSupportChunk, ...] = (),
    reviewer_actor: str,
    reviewed_at: datetime,
    store: SupportEvidenceStore,
) -> RunBoundSupportEvidence:
    """Apply one administrator decision after the candidate was re-read."""

    selected: list[VerifiedChunkChoice] = []
    for candidate in candidates:
        if not candidate.valid or candidate.chunk is None:
            raise SupportPreparationError("selected chunk failed server verification")
        if candidate.source_provenance_sha256 != evidence.source_provenance_sha256:
            raise SupportPreparationError("selected chunk provenance differs from the run-pinned corpus")
        selected.append(verified_chunk_choice(candidate, evidence.source_provenance_sha256))
    return store.review_evidence(
        evidence,
        selected_chunks=tuple(selected),
        reviewer_actor=reviewer_actor,
        reviewed_at=reviewed_at,
    )


def _review_evidence(
    evidence: RunBoundSupportEvidence,
    *,
    selected_chunks: tuple[VerifiedChunkChoice, ...],
    reviewer_actor: str,
    reviewed_at: datetime,
    save: Any,
) -> RunBoundSupportEvidence:
    if evidence.match_status in {"review_approved", "review_rejected"}:
        same = tuple(evidence.selected_chunks or ((evidence.selected_chunk,) if evidence.selected_chunk else ())) == selected_chunks
        if evidence.match_status == "review_rejected":
            same = not selected_chunks
        if not same:
            raise SupportPreparationError("support review is already bound to a different decision")
        return evidence
    if evidence.match_status == "eligible":
        if (
            evidence.binding_version != "run-bound-support-v2"
            or evidence.parent_id is None
            or evidence.parent_content_hash is None
            or evidence.parent_span_sha256 is None
        ):
            raise SupportPreparationError("eligible hierarchical support is missing its run-bound parent span")
        proposed = tuple(evidence.selected_chunks or ((evidence.selected_chunk,) if evidence.selected_chunk else ()))
        if selected_chunks and selected_chunks != proposed:
            raise SupportPreparationError("eligible support can only approve its complete server-proposed child set")
    if len({choice.chunk_id for choice in selected_chunks}) != len(selected_chunks):
        raise SupportPreparationError("selected chunks must have unique IDs")
    if any(choice.corpus_version_id != evidence.corpus_version_id for choice in selected_chunks):
        raise SupportPreparationError("selected chunk belongs to another corpus")
    if any(choice.chunk_id not in evidence.candidate_chunk_ids for choice in selected_chunks):
        raise SupportPreparationError("selected chunk is not a server-listed candidate")
    parent_bindings = {(choice.parent_id, choice.parent_content_hash) for choice in selected_chunks}
    if len(parent_bindings) > 1:
        raise SupportPreparationError("selected chunks must belong to one verified parent")
    if len(selected_chunks) > 1 and (None, None) in parent_bindings:
        raise SupportPreparationError("multiple selected chunks require hierarchical parent metadata")
    parent_id, parent_hash = next(iter(parent_bindings), (evidence.parent_id, evidence.parent_content_hash))
    if evidence.parent_id is not None and parent_bindings and (parent_id, parent_hash) != (
        evidence.parent_id, evidence.parent_content_hash
    ):
        raise SupportPreparationError("selected chunks do not match the bound parent")
    updated = RunBoundSupportEvidence.model_validate(
        evidence.model_dump()
        | {
            "match_status": "review_approved" if selected_chunks else "review_rejected",
            "verified_chunks": list(selected_chunks),
            "selected_chunk": selected_chunks[0] if len(selected_chunks) == 1 else None,
            "selected_chunks": list(selected_chunks),
            "parent_id": parent_id,
            "parent_content_hash": parent_hash,
            "binding_version": "run-bound-support-v2" if parent_id is not None else evidence.binding_version,
            "reviewer_actor": reviewer_actor,
            "reviewed_at": reviewed_at,
        }
    )
    save(updated)
    return updated


def _decide_support(
    evidence: RunBoundSupportEvidence,
    *,
    decision: Literal["approve", "reject"],
    reviewer_actor: str,
    reviewed_at: datetime,
    save: Any,
) -> RunBoundSupportEvidence:
    if evidence.match_status in {"review_approved", "review_rejected"}:
        raise SupportPreparationError("support review was already decided")
    if decision == "approve" and evidence.match_status == "eligible":
        selected = tuple(evidence.selected_chunks or ((evidence.selected_chunk,) if evidence.selected_chunk else ()))
        if not selected:
            raise SupportPreparationError("support proposal is incomplete")
        return _review_evidence(
            evidence,
            selected_chunks=selected,
            reviewer_actor=reviewer_actor,
            reviewed_at=reviewed_at,
            save=save,
        )
    if decision == "approve" and evidence.match_status == "inapplicable" and evidence.expected_refusal:
        updated = RunBoundSupportEvidence.model_validate(evidence.model_dump() | {
            "match_status": "review_approved", "reviewer_actor": reviewer_actor, "reviewed_at": reviewed_at,
        })
        save(updated)
        return updated
    if decision == "reject":
        return _review_evidence(
            evidence,
            selected_chunks=(),
            reviewer_actor=reviewer_actor,
            reviewed_at=reviewed_at,
            save=save,
        )
    raise SupportPreparationError("this support proposal cannot be approved")


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
        choice = verified_chunk_choice(record, source_provenance)
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
    candidate_ids = list(dict.fromkeys(candidates or []))
    if len(candidate_ids) > 64:
        reason = "support_spans_too_many_children"
        candidate_ids = []
    status = (
        "ambiguous" if reason == "ambiguous_support"
        else "inapplicable" if expected_refusal
        else "invalid" if reason in {
            "invalid_chunk_metadata", "placeholder_chunk_id", "invalid_provenance",
            "invalid_parent_child_hierarchy", "support_spans_too_many_children",
        }
        else "missing"
    )
    return RunBoundSupportEvidence(
        run_id=run.run_id,
        item_id=item_id,
        candidate_snapshot_sha256=snapshot_digest,
        corpus_version_id=run.corpus_version_id,
        source_provenance_sha256=provenance,
        match_status=status,
        candidate_chunk_ids=candidate_ids,
        reason=reason,
        expected_refusal=expected_refusal,
    )


def verified_chunk_choice(record: StoredSupportChunk, provenance: str) -> VerifiedChunkChoice:
    if record.chunk is None or not record.valid:
        raise SupportPreparationError("cannot verify an invalid chunk")
    if isinstance(record.chunk, ChildDocumentChunk):
        if record.parent is None:
            raise SupportPreparationError("child parent metadata is unavailable")
        return VerifiedChunkChoice(
            chunk_id=record.chunk.chunk_id,
            corpus_version_id=record.chunk.corpus_version_id,
            content_hash=record.chunk.content_hash,
            page=record.chunk.page_start,
            section=" / ".join(record.chunk.section_path),
            source_provenance_sha256=provenance,
            parent_id=record.parent.parent_id,
            parent_content_hash=record.parent.content_hash,
        )
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
    elif isinstance(chunk, ChildDocumentChunk) and (
        record.parent is None
        or record.parent.parent_id != chunk.parent_id
        or record.parent.corpus_version_id != corpus_version_id
        or sha256(record.parent.text.encode("utf-8")).hexdigest() != record.parent.content_hash
    ):
        reason = "child parent metadata is invalid"
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
