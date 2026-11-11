"""Durable, atomically activated evaluation dataset generations.

Dataset records are written before activation.  The single active-generation
pointer is the visibility boundary: incomplete staging can leave orphaned
documents, but it can never replace the currently visible dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Callable, Protocol, TypeVar

from google.cloud import firestore
from google.api_core.exceptions import AlreadyExists
from google.cloud.firestore_v1.base_query import FieldFilter

from app.contracts import (
    ActiveEvaluationGeneration,
    CandidateDatasetBinding,
    DatasetHead,
    EvaluationCandidateSnapshot,
    EvaluationRowV2,
    EvaluationRunSnapshot,
    ServingDatasetBinding,
    EvaluationDataset,
    EvaluationItem,
    EvaluationReportEligibility,
    GoldAnnotation,
    ChildDocumentChunk,
    CorpusVersion,
    ParentChunk,
)
from app.corpus.indexing import PARENT_COLLECTION
from app.corpus.versions import hierarchical_validation_evidence, validate_hierarchical_records
from app.evaluation.dataset import (
    CandidateBindingError,
    CandidateGoldBinding,
    CandidateGoldReview,
    GoldAnnotationRemap,
    candidate_review_fingerprint,
)


class EvaluationGenerationError(ValueError):
    """Base error for evaluation-generation persistence operations."""


class ConcurrentActivationError(EvaluationGenerationError):
    """Raised when the active generation changed before activation committed."""


class GenerationIntegrityError(EvaluationGenerationError):
    """Raised when a generation identifier is reused for different data."""


class ReviewStateError(EvaluationGenerationError):
    """Raised when an annotation review is not a permitted active-state transition."""


class _Snapshot(Protocol):
    exists: bool

    def to_dict(self) -> dict[str, Any]: ...


class _DocumentReference(Protocol):
    def get(self, *, transaction: Any | None = None) -> _Snapshot: ...

    def collection(self, collection_id: str) -> Any: ...

    def create(self, data: dict[str, Any]) -> Any: ...


class _FirestoreClient(Protocol):
    def collection(self, collection_id: str) -> Any: ...

    def batch(self) -> Any: ...

    def transaction(self) -> Any: ...


T = TypeVar("T")
_TransactionRunner = Callable[[Any, Callable[[Any], T]], T]


def _run_firestore_transaction(transaction: Any, callback: Callable[[Any], T]) -> T:
    """Run ``callback`` with Firestore's retrying transaction wrapper."""

    @firestore.transactional
    def run(active_transaction: Any) -> T:
        return callback(active_transaction)

    return run(transaction)


class FirestoreEvaluationGenerationStore:
    """Persist immutable generations and switch the active pointer atomically."""

    generations_collection_id = "evaluation_generations"
    active_pointer_collection_id = "evaluation_dataset_state"
    active_pointer_id = "active"
    _staging_batch_size = 400
    revision_collection_id = "evaluation_dataset_revisions_v2"
    corpus_heads_collection_id = "evaluation_dataset_heads_v2"
    revision_winners_collection_id = "evaluation_dataset_revision_winners_v2"
    run_snapshots_collection_id = "evaluation_run_snapshots_v2"

    def __init__(
        self,
        client: _FirestoreClient,
        *,
        transaction_runner: _TransactionRunner = _run_firestore_transaction,
    ) -> None:
        self.client = client
        self.generations = client.collection(self.generations_collection_id)
        self.pointer = client.collection(self.active_pointer_collection_id).document(self.active_pointer_id)
        self.revisions = client.collection(self.revision_collection_id)
        self.corpus_heads = client.collection(self.corpus_heads_collection_id)
        self.revision_winners = client.collection(self.revision_winners_collection_id)
        self.run_snapshots = client.collection(self.run_snapshots_collection_id)
        self._transaction_runner = transaction_runner

    def stage_revision(
        self, corpus_version_id: str, revision: int, rows: tuple[EvaluationRowV2, ...],
    ) -> DatasetHead:
        """Stage shape-validated rows without changing a visible pointer."""
        if not rows:
            raise EvaluationGenerationError("a revision requires at least one row")
        rows = tuple(EvaluationRowV2.model_validate(row.model_dump()) for row in rows)
        if len({row.item_id for row in rows}) != len(rows):
            raise EvaluationGenerationError("revision item IDs must be unique")
        if any(chunk_id.startswith(("imported-passage-", "placeholder-"))
               for row in rows for chunk_id in row.relevant_child_ids):
            raise EvaluationGenerationError("revision cannot contain legacy placeholder child IDs")
        payloads = [row.model_dump(mode="json") for row in rows]
        head = DatasetHead(corpus_version_id=corpus_version_id, revision=revision,
                           dataset_sha256=_revision_fingerprint(payloads))
        reference = self.revisions.document(_proposal_id(head))
        metadata = {**head.model_dump(mode="json"), "status": "staging", "item_count": len(rows)}
        try:
            reference.create(metadata)
        except AlreadyExists:
            existing = reference.get().to_dict()
            if any(existing.get(key) != value for key, value in head.model_dump(mode="json").items()):
                raise GenerationIntegrityError("revision is already bound to different data")
            if existing.get("status") == "staged":
                self.get_revision(head)
                return head
        writes = [
            (reference.collection("items").document(sha256(row.item_id.encode()).hexdigest()),
             {"ordinal": ordinal, "row": payload})
            for ordinal, (row, payload) in enumerate(zip(rows, payloads, strict=True))
        ]
        self._commit_staged_writes(writes)
        reference.update({"status": "staged"})
        self.get_revision(head)
        return head

    def get_revision(self, head: DatasetHead) -> tuple[EvaluationRowV2, ...]:
        reference = self.revisions.document(_proposal_id(head))
        snapshot = reference.get()
        if not snapshot.exists:
            raise EvaluationGenerationError("revision is unavailable")
        metadata = snapshot.to_dict()
        if metadata.get("status") != "staged" or any(
            metadata.get(key) != value for key, value in head.model_dump(mode="json").items()
        ):
            raise GenerationIntegrityError("revision metadata is inconsistent")
        documents = [item.to_dict() for item in reference.collection("items").stream()]
        documents.sort(key=lambda item: item["ordinal"])
        if [item["ordinal"] for item in documents] != list(range(metadata.get("item_count", -1))):
            raise GenerationIntegrityError("revision items are incomplete")
        payloads = [item["row"] for item in documents]
        if _revision_fingerprint(payloads) != head.dataset_sha256:
            raise GenerationIntegrityError("revision content digest does not match")
        return tuple(EvaluationRowV2.model_validate(payload) for payload in payloads)

    def corpus_head(self, corpus_version_id: str) -> DatasetHead | None:
        snapshot = self.corpus_heads.document(_corpus_id(corpus_version_id)).get()
        return DatasetHead.model_validate(snapshot.to_dict()) if snapshot.exists else None

    def activate_revision(self, head: DatasetHead, *, expected_revision: int | None) -> DatasetHead:
        self.get_revision(head)
        pointer = self.corpus_heads.document(_corpus_id(head.corpus_version_id))
        winner = self.revision_winners.document(_revision_id(head.corpus_version_id, head.revision))

        def apply(transaction: Any) -> DatasetHead:
            snapshot = pointer.get(transaction=transaction)
            winner_snapshot = winner.get(transaction=transaction)
            current = DatasetHead.model_validate(snapshot.to_dict()) if snapshot.exists else None
            if (current.revision if current else None) != expected_revision:
                raise ConcurrentActivationError("corpus dataset revision changed before activation")
            if head.revision != (current.revision + 1 if current else 1):
                raise EvaluationGenerationError("revision must advance the corpus head by one")
            if winner_snapshot.exists:
                raise ConcurrentActivationError("corpus dataset revision is already published")
            transaction.set(winner, head.model_dump(mode="json"))
            transaction.set(pointer, head.model_dump(mode="json"))
            return head

        return self._transaction_runner(self.client.transaction(), apply)

    def create_run_snapshot(
        self, *, run_id: str, binding: CandidateDatasetBinding | ServingDatasetBinding,
        configuration_version_id: str, configuration: EvaluationCandidateSnapshot,
        item_ids: tuple[str, ...] | None = None,
    ) -> EvaluationRunSnapshot:
        """Pin selected rows and the complete candidate configuration before dispatch."""
        if isinstance(binding, ServingDatasetBinding):
            raise EvaluationGenerationError("serving snapshots require authoritative release resolution")
        reference = self.run_snapshots.document(sha256(run_id.encode()).hexdigest())
        existing = reference.get()
        if existing.exists and existing.to_dict().get("status") == "ready":
            saved, saved_configuration = self.get_run_snapshot(run_id)
            selection = "full" if item_ids is None else "subset"
            if (saved.binding != binding or saved.configuration_version_id != configuration_version_id
                    or saved_configuration != configuration or saved.selection != selection
                    or (item_ids is not None and tuple(row.item_id for row in saved.items) != item_ids)):
                raise GenerationIntegrityError("run ID is already bound to different inputs")
            return saved
        winner = self.revision_winners.document(_revision_id(binding.corpus_version_id, binding.dataset_revision))
        revision = winner.get()
        if not revision.exists:
            raise EvaluationGenerationError("run requires a published corpus revision")
        head = DatasetHead.model_validate(revision.to_dict())
        if self.corpus_head(binding.corpus_version_id) != head:
            raise ConcurrentActivationError("corpus dataset revision is no longer current")
        if configuration.corpus_binding.corpus_version_id != binding.corpus_version_id:
            raise GenerationIntegrityError("run configuration belongs to a different corpus")
        rows = self.get_revision(head)
        selected = tuple(row.item_id for row in rows) if item_ids is None else item_ids
        available = {row.item_id: row for row in rows}
        if not selected or len(set(selected)) != len(selected) or any(item not in available for item in selected):
            raise EvaluationGenerationError("run selection must contain unique existing item IDs")
        selected_rows = tuple(available[item] for item in selected)
        values = {
            "run_id": run_id, "binding": binding.model_dump(mode="json"),
            "configuration_version_id": configuration_version_id,
            "selection": "full" if item_ids is None else "subset",
            "items": [row.model_dump(mode="json") for row in selected_rows],
        }
        configuration_payload = configuration.model_dump(mode="json")
        digest = _snapshot_fingerprint(values, configuration_payload)
        result = EvaluationRunSnapshot.model_validate({**values, "snapshot_sha256": digest})
        metadata = {key: value for key, value in result.model_dump(mode="json").items() if key != "items"}
        metadata.update({"configuration": configuration_payload, "status": "staging", "item_count": len(selected_rows)})
        try:
            reference.create(metadata)
        except AlreadyExists:
            existing = reference.get().to_dict()
            if existing.get("snapshot_sha256") != digest:
                raise GenerationIntegrityError("run ID is already bound to different inputs")
            if existing.get("status") == "ready":
                return self.get_run_snapshot(run_id)[0]
        writes = [
            (reference.collection("items").document(sha256(row.item_id.encode()).hexdigest()),
             {"ordinal": ordinal, "row": row.model_dump(mode="json")})
            for ordinal, row in enumerate(selected_rows)
        ]
        self._commit_staged_writes(writes)
        pointer = self.corpus_heads.document(_corpus_id(binding.corpus_version_id))

        def publish(transaction: Any) -> None:
            current = pointer.get(transaction=transaction)
            if not current.exists or DatasetHead.model_validate(current.to_dict()) != head:
                raise ConcurrentActivationError("corpus dataset revision changed during run preparation")
            transaction.update(reference, {"status": "ready"})

        self._transaction_runner(self.client.transaction(), publish)
        return self.get_run_snapshot(run_id)[0]

    def get_run_snapshot(self, run_id: str) -> tuple[EvaluationRunSnapshot, EvaluationCandidateSnapshot]:
        reference = self.run_snapshots.document(sha256(run_id.encode()).hexdigest())
        snapshot = reference.get()
        if not snapshot.exists or snapshot.to_dict().get("status") != "ready":
            raise EvaluationGenerationError("run snapshot is unavailable")
        metadata = snapshot.to_dict()
        documents = [item.to_dict() for item in reference.collection("items").stream()]
        documents.sort(key=lambda item: item["ordinal"])
        if [item["ordinal"] for item in documents] != list(range(metadata.get("item_count", -1))):
            raise GenerationIntegrityError("run snapshot items are incomplete")
        values = {key: metadata[key] for key in ("run_id", "binding", "configuration_version_id", "selection")}
        values["items"] = [item["row"] for item in documents]
        if values["run_id"] != run_id or _snapshot_fingerprint(values, metadata["configuration"]) != metadata["snapshot_sha256"]:
            raise GenerationIntegrityError("run snapshot digest does not match")
        result = EvaluationRunSnapshot.model_validate({**values, "snapshot_sha256": metadata["snapshot_sha256"]})
        return result, EvaluationCandidateSnapshot.model_validate(metadata["configuration"])

    def stage(self, dataset: EvaluationDataset, *, candidate_review: CandidateGoldReview | None = None) -> str:
        """Durably write one immutable generation without changing visibility."""

        if candidate_review is not None:
            annotations = [
                annotation.model_copy(
                    update={
                        "relevant_chunk_ids": list(candidate_review.remap_for(annotation.item_id).relevant_chunk_ids),
                        "review_status": "candidate",
                        "semantic_assessment": None,
                    }
                )
                for annotation in dataset.gold_annotations
            ]
            dataset = dataset.model_copy(update={"gold_annotations": annotations}, deep=True)
        generation = self.generations.document(dataset.dataset_version_id)
        _validate_candidate_review(dataset, candidate_review)
        fingerprint = _dataset_fingerprint(dataset, candidate_review)
        payload = _generation_payload(dataset, fingerprint, status="staging", candidate_review=candidate_review)
        existing = generation.get()
        if existing.exists:
            existing_payload = existing.to_dict()
            if existing_payload.get("content_sha256") != fingerprint:
                raise GenerationIntegrityError("generation_id is already bound to different evaluation data")
            if existing_payload.get("status") != "staging":
                return dataset.dataset_version_id
        else:
            try:
                generation.create(payload)
            except AlreadyExists:
                # A competing writer claimed this identifier.  Re-read it
                # before writing any records so immutable data cannot mix.
                existing_payload = generation.get().to_dict()
                if existing_payload.get("content_sha256") != fingerprint:
                    raise GenerationIntegrityError("generation_id is already bound to different evaluation data")
                if existing_payload.get("status") != "staging":
                    return dataset.dataset_version_id

        writes: list[tuple[Any, dict[str, Any]]] = []
        for item in dataset.items:
            writes.append((generation.collection("items").document(item.item_id), item.model_dump(mode="json")))
        for annotation in dataset.gold_annotations:
            annotation_payload = annotation.model_dump(mode="json")
            if candidate_review is not None:
                annotation_payload["candidate_review"] = candidate_review.remap_for(annotation.item_id).model_dump()
            writes.append(
                (
                    generation.collection("annotations").document(annotation.item_id),
                    annotation_payload,
                )
            )
        self._commit_staged_writes(writes)

        generation.update({"status": "staged"})
        return dataset.dataset_version_id

    def activate(
        self,
        generation_id: str,
        *,
        expected_active_generation_id: str | None,
    ) -> ActiveEvaluationGeneration:
        """Compare-and-set the active pointer after staging has completed."""

        generation = self.generations.document(generation_id)

        def apply(transaction: Any) -> ActiveEvaluationGeneration:
            pointer_snapshot = self.pointer.get(transaction=transaction)
            current_id = pointer_snapshot.to_dict().get("generation_id") if pointer_snapshot.exists else None
            if current_id != expected_active_generation_id:
                raise ConcurrentActivationError("active generation changed before activation could commit")

            staged_snapshot = generation.get(transaction=transaction)
            if not staged_snapshot.exists:
                raise EvaluationGenerationError(f"unknown evaluation generation: {generation_id}")
            staged = staged_snapshot.to_dict()
            if staged.get("status") not in {"staged", "active"}:
                raise EvaluationGenerationError("only a staged generation can become active")

            if current_id == generation_id:
                return _active_generation(staged)

            if current_id is not None:
                transaction.update(self.generations.document(current_id), {"status": "superseded"})
            transaction.update(generation, {"status": "active"})
            transaction.set(
                self.pointer,
                {"generation_id": generation_id, "activated_at": datetime.now(timezone.utc).isoformat()},
            )
            return _active_generation(staged)

        return self._transaction_runner(self.client.transaction(), apply)

    def active(self) -> ActiveEvaluationGeneration | None:
        """Resolve the only generation visible to current-evidence queries."""

        pointer_snapshot = self.pointer.get()
        if not pointer_snapshot.exists:
            return None
        generation_id = pointer_snapshot.to_dict().get("generation_id")
        if not isinstance(generation_id, str):
            raise EvaluationGenerationError("active-generation pointer is invalid")
        snapshot = self.generations.document(generation_id).get()
        if not snapshot.exists:
            raise EvaluationGenerationError("active-generation pointer references missing data")
        return _active_generation(snapshot.to_dict())

    def get(self, generation_id: str) -> EvaluationDataset | None:
        """Load one immutable evaluator-authored dataset generation."""

        generation = self.generations.document(generation_id)
        snapshot = generation.get()
        if not snapshot.exists:
            return None
        items = sorted(
            (EvaluationItem.model_validate(item.to_dict()) for item in generation.collection("items").stream()),
            key=lambda item: item.item_id,
        )
        if not items:
            raise EvaluationGenerationError("evaluation generation has no staged items")
        annotations = sorted(
            (_annotation_from_payload(annotation.to_dict()) for annotation in generation.collection("annotations").stream()),
            key=lambda annotation: annotation.item_id,
        )
        return EvaluationDataset(
            dataset_version_id=generation_id,
            items=items,
            gold_annotations=annotations,
        )

    def active_dataset(self) -> EvaluationDataset | None:
        """Resolve the active dataset immediately before an evaluation launch."""

        active = self.active()
        return self.get(active.generation_id) if active is not None else None

    def report_eligibility(self, generation_id: str) -> EvaluationReportEligibility:
        """Classify evidence from one consistent active-pointer snapshot."""

        generation_ref = self.generations.document(generation_id)

        def apply(transaction: Any) -> EvaluationReportEligibility:
            snapshot = generation_ref.get(transaction=transaction)
            if not snapshot.exists:
                raise EvaluationGenerationError(f"unknown evaluation generation: {generation_id}")
            pointer_snapshot = self.pointer.get(transaction=transaction)
            active_id = pointer_snapshot.to_dict().get("generation_id") if pointer_snapshot.exists else None
            generation = snapshot.to_dict()
            is_current = active_id == generation_id
            approved_gold_count = int(generation.get("approved_gold_count", 0))
            release_eligible = is_current and 30 <= approved_gold_count <= 50
            return EvaluationReportEligibility(
                generation_id=generation_id,
                eligibility="release_evidence" if release_eligible else "exploratory",
                approved_gold_count=approved_gold_count,
                is_current_generation=is_current,
                is_superseded=not is_current,
            )

        return self._transaction_runner(self.client.transaction(), apply)

    def candidate_binding(self, generation_id: str) -> CandidateGoldBinding | None:
        """Return the immutable candidate binding recorded with a generation."""

        payload = self.generations.document(generation_id).get().to_dict()
        review = payload.get("candidate_review")
        binding = review.get("binding") if isinstance(review, dict) else None
        if not isinstance(binding, dict):
            return None
        try:
            return CandidateGoldBinding(**binding)
        except (TypeError, CandidateBindingError):
            raise GenerationIntegrityError("candidate binding is invalid") from None

    def get_candidate_chunk(self, *, corpus_version_id: str, chunk_id: str):
        """Resolve one persisted vector document without crossing corpus versions."""

        snapshot = self.client.collection("document_chunks_v2").document(chunk_id).get()
        if not snapshot.exists:
            return None
        payload = snapshot.to_dict()
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("corpus_version_id") != corpus_version_id:
            return None
        from app.contracts import DocumentChunk

        try:
            return DocumentChunk(
                chunk_id=str(metadata["chunk_id"]),
                corpus_version_id=str(metadata["corpus_version_id"]),
                page=int(metadata["page"]),
                section=str(metadata["section"]),
                chunk_ordinal=int(metadata["chunk_ordinal"]),
                content_hash=str(metadata["content_hash"]),
                text=str(payload["text"]),
                embedding=list(payload["embedding"]),
            )
        except (KeyError, TypeError, ValueError):
            raise GenerationIntegrityError("candidate chunk is invalid") from None

    def get_parse_derived_records(
        self, *, corpus_version_id: str
    ) -> tuple[tuple[ParentChunk, ...], tuple[ChildDocumentChunk, ...]] | None:
        """Load a complete, ready hierarchy bound to the selected corpus version."""
        version_snapshot = self.client.collection("corpus_versions").document(corpus_version_id).get()
        if not version_snapshot.exists:
            return None
        try:
            version = CorpusVersion.model_validate(version_snapshot.to_dict())
        except (TypeError, ValueError):
            raise GenerationIntegrityError("selected corpus version is invalid") from None
        evidence = version.validation_evidence
        binding = version.serving_binding
        if (
            version.version_id != corpus_version_id
            or version.status not in {"validated", "active"}
            or binding is None
            or binding.corpus_version_id != corpus_version_id
            or evidence is None
            or not evidence.index_ready
            or not evidence.filter_ready
        ):
            return None

        parent_records: list[ParentChunk] = []
        parent_rows = self.client.collection(PARENT_COLLECTION).where(
            filter=FieldFilter("corpus_version_id", "==", corpus_version_id)
        ).stream()
        for snapshot in parent_rows:
            payload = snapshot.to_dict()
            if not isinstance(payload, dict):
                raise GenerationIntegrityError("selected corpus parent is invalid")
            try:
                parent_records.append(ParentChunk.model_validate({
                    key: value for key, value in payload.items() if key != "metadata"
                }))
            except (TypeError, ValueError):
                raise GenerationIntegrityError("selected corpus parent provenance is invalid") from None

        child_records: list[ChildDocumentChunk] = []
        child_rows = self.client.collection(binding.vector_collection).where(
            filter=FieldFilter("metadata.corpus_version_id", "==", corpus_version_id)
        ).stream()
        for snapshot in child_rows:
            payload = snapshot.to_dict()
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if not isinstance(metadata, dict) or not {
                "schema_version", "chunk_id", "corpus_version_id", "parent_id", "section_path",
                "page_start", "page_end", "source_spans", "child_ordinal", "content_hash",
                "token_count", "corpus_manifest_sha256", "derivation_sha256",
            }.issubset(metadata):
                raise GenerationIntegrityError("selected corpus child lacks bound provenance")
            if (
                metadata.get("corpus_manifest_sha256") != binding.ingestion_fingerprint
                or metadata.get("derivation_sha256") != binding.configuration_fingerprint
            ):
                raise GenerationIntegrityError("selected corpus child has a conflicting corpus binding")
            try:
                child = ChildDocumentChunk.model_validate({
                    **{key: metadata[key] for key in (
                        "schema_version", "chunk_id", "corpus_version_id", "parent_id", "section_path",
                        "page_start", "page_end", "source_spans", "source_text", "child_ordinal",
                        "content_hash", "token_count",
                    ) if key in metadata},
                    "text": payload["text"],
                })
            except (KeyError, TypeError, ValueError):
                raise GenerationIntegrityError("selected corpus child provenance is invalid") from None
            snapshot_id = getattr(snapshot, "id", child.chunk_id)
            if snapshot_id != child.chunk_id:
                raise GenerationIntegrityError("selected corpus child ID is not deterministic")
            child_records.append(child)

        parents = tuple(parent_records)
        children = tuple(child_records)
        try:
            validate_hierarchical_records(corpus_version_id, parents, children)
            recomputed = hierarchical_validation_evidence(
                version, parents, children, index_ready=True, filter_ready=True
            )
        except (TypeError, ValueError):
            raise GenerationIntegrityError("selected corpus hierarchy is incomplete") from None
        if recomputed != evidence:
            raise GenerationIntegrityError("selected corpus hierarchy differs from readiness evidence")
        return parents, children

    def bind_candidate_review(
        self,
        generation_id: str,
        item_id: str,
        candidate_review: CandidateGoldReview,
    ) -> GoldAnnotation:
        """Atomically persist one candidate binding while leaving approval unchanged."""

        generation = self.generations.document(generation_id)
        annotation = generation.collection("annotations").document(item_id)

        def apply(transaction: Any) -> GoldAnnotation:
            pointer_snapshot = self.pointer.get(transaction=transaction)
            active_id = pointer_snapshot.to_dict().get("generation_id") if pointer_snapshot.exists else None
            if active_id != generation_id:
                raise ReviewStateError("candidate bindings are allowed only for the active evaluation generation")
            generation_snapshot = generation.get(transaction=transaction)
            annotation_snapshot = annotation.get(transaction=transaction)
            if not generation_snapshot.exists or not annotation_snapshot.exists:
                raise EvaluationGenerationError("unknown evaluation binding target")
            current_payload = annotation_snapshot.to_dict()
            current = _annotation_from_payload(current_payload)
            if current.review_status != "candidate":
                raise ReviewStateError("candidate binding requires a candidate annotation")

            generation_payload = generation_snapshot.to_dict()
            existing = generation_payload.get("candidate_review")
            existing_review = _candidate_review_from_payload(existing) if isinstance(existing, dict) else None
            remap = candidate_review.remap_for(item_id)
            effective_review = candidate_review
            if existing_review is not None:
                if existing_review.binding != candidate_review.binding or existing_review.expected_refusal != candidate_review.expected_refusal:
                    raise ReviewStateError("candidate binding is already set to different data")
                try:
                    existing_remap = existing_review.remap_for(item_id)
                except CandidateBindingError:
                    existing_remap = None
                if existing_remap is not None:
                    if existing_remap != remap:
                        raise ReviewStateError("candidate binding is already set to different data")
                    return current
                effective_review = CandidateGoldReview(
                    binding=candidate_review.binding,
                    remaps=existing_review.remaps + (remap,),
                    binding_version=candidate_review.binding_version,
                    expected_refusal=candidate_review.expected_refusal,
                    reviewed_by=candidate_review.reviewed_by,
                    reviewed_at=candidate_review.reviewed_at,
                    chunk_hashes={**(existing_review.chunk_hashes or {}), **(candidate_review.chunk_hashes or {})},
                )
            updated = current.model_copy(update={"relevant_chunk_ids": list(remap.relevant_chunk_ids)})
            stored_review = _candidate_review_from_payload(effective_review.model_dump())
            stored_review = CandidateGoldReview(
                binding=stored_review.binding,
                remaps=stored_review.remaps,
                binding_version=stored_review.binding_version,
                expected_refusal=stored_review.expected_refusal,
                reviewed_by=effective_review.reviewed_by,
                reviewed_at=effective_review.reviewed_at,
                chunk_hashes=stored_review.chunk_hashes,
            )
            transaction.update(
                annotation,
                {
                    **updated.model_dump(mode="json"),
                    "candidate_review": {
                        **remap.model_dump(),
                        "expected_refusal": effective_review.expected_refusal,
                        "binding_version": effective_review.binding_version,
                        "reviewed_by": effective_review.reviewed_by,
                        "reviewed_at": effective_review.reviewed_at,
                        "chunk_hashes": dict(effective_review.chunk_hashes or {}),
                    },
                },
            )
            transaction.update(
                generation,
                {
                    "candidate_review": stored_review.model_dump(),
                    "candidate_review_sha256": candidate_review_fingerprint(stored_review),
                },
            )
            return updated

        return self._transaction_runner(self.client.transaction(), apply)

    def generation_status(self, generation_id: str) -> str | None:
        """Return a persisted lifecycle status for audit and reproduction queries."""

        snapshot = self.generations.document(generation_id).get()
        return snapshot.to_dict().get("status") if snapshot.exists else None

    def review(self, generation_id: str, item_id: str, review_status: str) -> GoldAnnotation:
        """Persist one candidate review, but only on the current generation."""

        if review_status not in {"approved", "rejected"}:
            raise ReviewStateError("review status must be approved or rejected")

        generation = self.generations.document(generation_id)
        annotation = generation.collection("annotations").document(item_id)

        def apply(transaction: Any) -> GoldAnnotation:
            pointer_snapshot = self.pointer.get(transaction=transaction)
            active_id = pointer_snapshot.to_dict().get("generation_id") if pointer_snapshot.exists else None
            if active_id != generation_id:
                raise ReviewStateError("reviews are allowed only for the active evaluation generation")

            generation_snapshot = generation.get(transaction=transaction)
            if not generation_snapshot.exists:
                raise EvaluationGenerationError(f"unknown evaluation generation: {generation_id}")
            annotation_snapshot = annotation.get(transaction=transaction)
            if not annotation_snapshot.exists:
                raise EvaluationGenerationError(f"unknown annotation item: {item_id}")
            annotation_payload = annotation_snapshot.to_dict()
            current = _annotation_from_payload(annotation_payload)
            if current.review_status != "candidate":
                raise ReviewStateError("annotation review is already final")
            candidate_review = annotation_payload.get("candidate_review")
            if review_status == "approved" and isinstance(candidate_review, dict):
                if candidate_review.get("remap_status") != "matched" and not (
                    candidate_review.get("remap_status") == "refusal"
                    and candidate_review.get("expected_refusal") is True
                ):
                    raise ReviewStateError("unresolved candidate support cannot be approved")

            updated = current.model_copy(update={"review_status": review_status})
            approved_count = int(generation_snapshot.to_dict().get("approved_gold_count", 0))
            transaction.update(annotation, updated.model_dump(mode="json"))
            transaction.update(generation, {"approved_gold_count": approved_count + (review_status == "approved")})
            return updated

        return self._transaction_runner(self.client.transaction(), apply)

    def _commit_staged_writes(self, writes: list[tuple[Any, dict[str, Any]]]) -> None:
        for start in range(0, len(writes), self._staging_batch_size):
            batch = self.client.batch()
            for reference, payload in writes[start : start + self._staging_batch_size]:
                batch.set(reference, payload)
            batch.commit()


def _corpus_id(corpus_version_id: str) -> str:
    return sha256(corpus_version_id.encode()).hexdigest()


def _revision_id(corpus_version_id: str, revision: int) -> str:
    return f"{_corpus_id(corpus_version_id)}-{revision}"


def _proposal_id(head: DatasetHead) -> str:
    return f"{_revision_id(head.corpus_version_id, head.revision)}-{head.dataset_sha256}"


def _revision_fingerprint(payloads: list[dict[str, Any]]) -> str:
    return sha256(json.dumps(payloads, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _snapshot_fingerprint(values: dict[str, Any], configuration: dict[str, Any]) -> str:
    payload = {"snapshot": values, "configuration": configuration}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _dataset_fingerprint(dataset: EvaluationDataset, candidate_review: CandidateGoldReview | None = None) -> str:
    content: dict[str, object] = {"dataset": dataset.model_dump(mode="json")}
    if candidate_review is not None:
        content["candidate_review"] = candidate_review.model_dump()
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _generation_payload(
    dataset: EvaluationDataset,
    fingerprint: str,
    *,
    status: str,
    candidate_review: CandidateGoldReview | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generation_id": dataset.dataset_version_id,
        "item_count": len(dataset.items),
        "approved_gold_count": sum(
            annotation.review_status == "approved" for annotation in dataset.gold_annotations
        ),
        "status": status,
        "content_sha256": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if candidate_review is not None:
        payload["candidate_review"] = candidate_review.model_dump()
        payload["candidate_review_sha256"] = candidate_review_fingerprint(candidate_review)
    return payload


def _validate_candidate_review(
    dataset: EvaluationDataset, candidate_review: CandidateGoldReview | None
) -> None:
    if candidate_review is None:
        return
    annotation_ids = {annotation.item_id for annotation in dataset.gold_annotations}
    remap_ids = {remap.item_id for remap in candidate_review.remaps}
    if annotation_ids != remap_ids:
        raise GenerationIntegrityError("candidate review must bind every and only dataset annotation")
    if any(annotation.review_status != "candidate" for annotation in dataset.gold_annotations):
        raise GenerationIntegrityError("candidate review cannot carry forward annotation approval")
    for remap in candidate_review.remaps:
        if any(chunk_id.startswith(("imported-passage-", "placeholder-")) for chunk_id in remap.relevant_chunk_ids):
            raise GenerationIntegrityError("candidate review cannot retain placeholder chunk IDs")
        if remap.remap_status == "matched" and any(
            (candidate_review.chunk_hashes or {}).get(chunk_id) is None for chunk_id in remap.relevant_chunk_ids
        ):
            raise GenerationIntegrityError("matched candidate chunks require content hashes")
        if (remap.parent_id is None) != (remap.parent_content_hash is None):
            raise GenerationIntegrityError("candidate parent identity and hash must be supplied together")


def _active_generation(payload: dict[str, Any]) -> ActiveEvaluationGeneration:
    if not isinstance(payload.get("generation_id"), str):
        raise EvaluationGenerationError("evaluation generation is missing its generation_id")
    return ActiveEvaluationGeneration(
        generation_id=payload["generation_id"],
        item_count=payload["item_count"],
        approved_gold_count=payload["approved_gold_count"],
        status="active",
    )


def _annotation_from_payload(payload: dict[str, Any]) -> GoldAnnotation:
    """Remove durable remap metadata before validating the public annotation contract."""

    return GoldAnnotation.model_validate({key: value for key, value in payload.items() if key != "candidate_review"})


def _candidate_review_from_payload(payload: dict[str, Any]) -> CandidateGoldReview:
    try:
        binding = CandidateGoldBinding(**payload["binding"])
        remaps = tuple(
            GoldAnnotationRemap(**{key: value for key, value in remap.items() if key != "review_status"})
            for remap in payload["remaps"]
        )
        return CandidateGoldReview(
            binding=binding,
            remaps=remaps,
            binding_version=payload.get("binding_version", "candidate-binding-v1"),
            expected_refusal=bool(payload.get("expected_refusal", False)),
            reviewed_by=payload.get("reviewed_by"),
            reviewed_at=payload.get("reviewed_at"),
            chunk_hashes=payload.get("chunk_hashes") or {},
        )
    except (KeyError, TypeError, ValueError, CandidateBindingError):
        raise GenerationIntegrityError("candidate binding is invalid") from None
