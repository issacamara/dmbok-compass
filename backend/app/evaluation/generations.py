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

from app.contracts import ActiveEvaluationGeneration, EvaluationDataset


class EvaluationGenerationError(ValueError):
    """Base error for evaluation-generation persistence operations."""


class ConcurrentActivationError(EvaluationGenerationError):
    """Raised when the active generation changed before activation committed."""


class GenerationIntegrityError(EvaluationGenerationError):
    """Raised when a generation identifier is reused for different data."""


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

    def __init__(
        self,
        client: _FirestoreClient,
        *,
        transaction_runner: _TransactionRunner = _run_firestore_transaction,
    ) -> None:
        self.client = client
        self.generations = client.collection(self.generations_collection_id)
        self.pointer = client.collection(self.active_pointer_collection_id).document(self.active_pointer_id)
        self._transaction_runner = transaction_runner

    def stage(self, dataset: EvaluationDataset) -> str:
        """Durably write one immutable generation without changing visibility."""

        generation = self.generations.document(dataset.dataset_version_id)
        fingerprint = _dataset_fingerprint(dataset)
        payload = _generation_payload(dataset, fingerprint, status="staging")
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
            writes.append(
                (
                    generation.collection("annotations").document(annotation.item_id),
                    annotation.model_dump(mode="json"),
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

    def generation_status(self, generation_id: str) -> str | None:
        """Return a persisted lifecycle status for audit and reproduction queries."""

        snapshot = self.generations.document(generation_id).get()
        return snapshot.to_dict().get("status") if snapshot.exists else None

    def _commit_staged_writes(self, writes: list[tuple[Any, dict[str, Any]]]) -> None:
        for start in range(0, len(writes), self._staging_batch_size):
            batch = self.client.batch()
            for reference, payload in writes[start : start + self._staging_batch_size]:
                batch.set(reference, payload)
            batch.commit()


def _dataset_fingerprint(dataset: EvaluationDataset) -> str:
    encoded = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _generation_payload(dataset: EvaluationDataset, fingerprint: str, *, status: str) -> dict[str, Any]:
    return {
        "generation_id": dataset.dataset_version_id,
        "item_count": len(dataset.items),
        "approved_gold_count": sum(
            annotation.review_status == "approved" for annotation in dataset.gold_annotations
        ),
        "status": status,
        "content_sha256": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _active_generation(payload: dict[str, Any]) -> ActiveEvaluationGeneration:
    if not isinstance(payload.get("generation_id"), str):
        raise EvaluationGenerationError("evaluation generation is missing its generation_id")
    return ActiveEvaluationGeneration(
        generation_id=payload["generation_id"],
        item_count=payload["item_count"],
        approved_gold_count=payload["approved_gold_count"],
        status="active",
    )
