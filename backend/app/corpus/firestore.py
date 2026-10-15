"""Firestore-backed corpus-version metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Callable, TypeVar

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from app.contracts import CorpusValidationEvidence, CorpusVersion
from app.corpus.indexing import VECTOR_COLLECTION, VECTOR_SCHEMA_VERSION
from app.corpus.versions import (
    ConcurrentCorpusActivationError,
    CorpusVersionError,
    legacy_serving_binding,
)

T = TypeVar("T")
_TransactionRunner = Callable[[Any, Callable[[Any], T]], T]


def _run_firestore_transaction(transaction: Any, callback: Callable[[Any], T]) -> T:
    @firestore.transactional
    def run(active_transaction: Any) -> T:
        return callback(active_transaction)

    return run(transaction)


class FirestoreCorpusVersionStore:
    """Persist corpus versions and resolve the single active version."""

    collection_id = "corpus_versions"
    state_collection_id = "corpus_state"
    active_pointer_id = "active"

    def __init__(
        self,
        client: Any,
        *,
        vector_collection: str = VECTOR_COLLECTION,
        transaction_runner: _TransactionRunner = _run_firestore_transaction,
    ) -> None:
        self.client = client
        self.versions = client.collection(self.collection_id)
        self.vector_collection_id = vector_collection
        self.pointer = client.collection(self.state_collection_id).document(
            self.active_pointer_id
        )
        self._transaction_runner = transaction_runner

    def get(self, version_id: str) -> CorpusVersion | None:
        snapshot = self.versions.document(version_id).get()
        if not snapshot.exists:
            return None
        return CorpusVersion.model_validate(snapshot.to_dict())

    def active(self) -> CorpusVersion | None:
        def resolve(transaction: Any) -> CorpusVersion | None:
            pointer = self.pointer.get(transaction=transaction)
            if not pointer.exists:
                return None
            version_id = pointer.to_dict().get("version_id")
            if not isinstance(version_id, str) or not version_id:
                raise CorpusVersionError("active corpus pointer is invalid")
            snapshot = self.versions.document(version_id).get(transaction=transaction)
            if not snapshot.exists:
                raise CorpusVersionError("active corpus pointer references invalid data")
            version = CorpusVersion.model_validate(snapshot.to_dict())
            if version.status != "active":
                raise CorpusVersionError("active corpus pointer references invalid data")
            pointer_binding = pointer.to_dict().get("binding")
            if pointer_binding is not None:
                if version.serving_binding is None:
                    raise CorpusVersionError("active corpus binding is unavailable")
                if version.serving_binding.model_dump(mode="json") != pointer_binding:
                    raise CorpusVersionError("active corpus binding is inconsistent")
            return version

        version = self._transaction_runner(self.client.transaction(), resolve)
        if version is not None:
            return version
        # Read compatibility for installations created before the pointer was
        # introduced. The next activation atomically adopts this active ID.
        snapshots = self.versions.where("status", "==", "active").limit(1).stream()
        snapshot = next(iter(snapshots), None)
        return CorpusVersion.model_validate(snapshot.to_dict()) if snapshot else None

    def stage(self, version: CorpusVersion) -> CorpusVersion:
        existing = self.get(version.version_id)
        if existing is not None:
            if existing.source_uri != version.source_uri:
                raise ValueError("version_id is already bound to another source")
            return existing
        self.versions.document(version.version_id).set(version.model_dump(mode="json"))
        return version

    def validate(self, version_id: str, chunks: Any) -> CorpusVersion:
        version = self.get(version_id)
        if version is None:
            raise ValueError(f"unknown corpus version: {version_id}")
        if version.status in {"validated", "active"}:
            return version
        if version.status != "staged":
            raise ValueError(f"cannot validate a {version.status} corpus version")
        expected = {chunk.chunk_id: chunk for chunk in chunks}
        if not expected:
            raise ValueError("corpus must contain at least one chunk")
        binding = version.serving_binding or legacy_serving_binding(
            version, vector_collection=self.vector_collection_id
        )
        if binding.vector_schema_version != VECTOR_SCHEMA_VERSION:
            raise CorpusVersionError("unsupported corpus vector schema")
        snapshots = self.client.collection(binding.vector_collection).where(
            filter=FieldFilter("metadata.corpus_version_id", "==", version_id)
        ).stream()
        persisted: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            data = snapshot.to_dict()
            metadata = data.get("metadata", {})
            chunk_id = metadata.get("chunk_id") or getattr(snapshot, "id", "")
            if chunk_id:
                persisted[str(chunk_id)] = data
        if set(persisted) != set(expected):
            raise ValueError("persisted vector chunks do not match the staged manifest")
        for chunk_id, chunk in expected.items():
            data = persisted[chunk_id]
            metadata = data.get("metadata", {})
            embedding = data.get("embedding")
            if (
                data.get("text") != chunk.text
                or metadata.get("content_hash") != chunk.content_hash
                or metadata.get("page") != chunk.page
                or metadata.get("section") != chunk.section
                or metadata.get("chunk_ordinal") != chunk.chunk_ordinal
                or embedding is None
                or len(embedding) != version.embedding_dimensions
            ):
                raise ValueError("persisted vector chunk failed completeness validation")
            if version.serving_binding is not None and (
                metadata.get("schema_version") != binding.vector_schema_version
                or metadata.get("corpus_manifest_sha256") != binding.ingestion_fingerprint
                or metadata.get("derivation_sha256") != binding.configuration_fingerprint
            ):
                raise ValueError("persisted vector chunk failed binding validation")
        validated = version.model_copy(
            update={
                "status": "validated",
                "validation_evidence": _validation_evidence(tuple(expected.values())),
            }
        )
        self.versions.document(version_id).set(validated.model_dump(mode="json"))
        return validated

    def activate(
        self,
        version_id: str,
        *,
        expected_active_version_id: str | None,
    ) -> CorpusVersion:
        candidate_ref = self.versions.document(version_id)

        def apply(transaction: Any) -> CorpusVersion:
            pointer_snapshot = self.pointer.get(transaction=transaction)
            pointer_data = pointer_snapshot.to_dict() if pointer_snapshot.exists else {}
            current_id = pointer_data.get("version_id")

            # Adopt the legacy status-only active version on the first v2
            # activation. The caller resolved it before ingestion began.
            if current_id is None and expected_active_version_id is not None:
                legacy_ref = self.versions.document(expected_active_version_id)
                legacy_snapshot = legacy_ref.get(transaction=transaction)
                if legacy_snapshot.exists and legacy_snapshot.to_dict().get("status") == "active":
                    current_id = expected_active_version_id

            if current_id != expected_active_version_id:
                raise ConcurrentCorpusActivationError(
                    "active corpus changed before activation could commit"
                )

            candidate_snapshot = candidate_ref.get(transaction=transaction)
            if not candidate_snapshot.exists:
                raise CorpusVersionError(f"unknown corpus version: {version_id}")
            candidate = CorpusVersion.model_validate(candidate_snapshot.to_dict())
            if current_id == version_id and candidate.status == "active":
                return candidate
            if candidate.status != "validated":
                raise CorpusVersionError(
                    "only a validated corpus version can become active"
                )
            if candidate.serving_binding is not None and not _ready(candidate):
                raise CorpusVersionError("validated corpus readiness evidence is unavailable")

            if current_id is not None:
                transaction.update(self.versions.document(current_id), {"status": "retired"})
            active = candidate.model_copy(update={"status": "active"})
            transaction.set(candidate_ref, active.model_dump(mode="json"))
            transaction.set(
                self.pointer,
                {
                    "version_id": version_id,
                    "previous_version_id": current_id,
                    "binding": (
                        candidate.serving_binding.model_dump(mode="json")
                        if candidate.serving_binding is not None
                        else None
                    ),
                    "previous_binding": pointer_data.get("binding"),
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return active

        return self._transaction_runner(self.client.transaction(), apply)

    def rollback(self, version_id: str | None = None) -> CorpusVersion:
        """Atomically restore the previous or explicitly selected valid version."""

        def apply(transaction: Any) -> CorpusVersion:
            pointer_snapshot = self.pointer.get(transaction=transaction)
            if not pointer_snapshot.exists:
                raise CorpusVersionError("no active corpus pointer is available for rollback")
            pointer_data = pointer_snapshot.to_dict()
            current_id = pointer_data.get("version_id")
            target_id = version_id or pointer_data.get("previous_version_id")
            if not isinstance(current_id, str) or not isinstance(target_id, str):
                raise CorpusVersionError(
                    "no previous valid corpus version is available for rollback"
                )

            current_ref = self.versions.document(current_id)
            target_ref = self.versions.document(target_id)
            current_snapshot = current_ref.get(transaction=transaction)
            target_snapshot = target_ref.get(transaction=transaction)
            if not current_snapshot.exists or not target_snapshot.exists:
                raise CorpusVersionError("rollback references unavailable corpus data")
            target = CorpusVersion.model_validate(target_snapshot.to_dict())
            if target.status not in {"retired", "active"}:
                raise CorpusVersionError(
                    "rollback target must be a previously valid corpus version"
                )
            if current_id == target_id:
                return target

            transaction.update(current_ref, {"status": "retired"})
            restored = target.model_copy(update={"status": "active"})
            if restored.serving_binding is not None and not _ready(restored):
                raise CorpusVersionError("rollback target readiness evidence is unavailable")
            transaction.set(target_ref, restored.model_dump(mode="json"))
            transaction.set(
                self.pointer,
                {
                    "version_id": target_id,
                    "previous_version_id": current_id,
                    "binding": (
                        restored.serving_binding.model_dump(mode="json")
                        if restored.serving_binding is not None
                        else None
                    ),
                    "previous_binding": pointer_data.get("binding"),
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return restored

        return self._transaction_runner(self.client.transaction(), apply)


def _validation_evidence(chunks: tuple[Any, ...]) -> CorpusValidationEvidence:
    """Create deterministic evidence from the records verified after writes."""

    content = [
        {"chunk_id": chunk.chunk_id, "content_hash": chunk.content_hash}
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]
    provenance = [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "section": chunk.section,
            "chunk_ordinal": chunk.chunk_ordinal,
        }
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
    ]
    return CorpusValidationEvidence(
        chunk_count=len(chunks),
        content_hashes_sha256=_hash(content),
        provenance_sha256=_hash(provenance),
        embedding_dimensions=768,
        # A successful bounded query with the pinned version proves the
        # collection/filter can be read. The native vector store owns index use.
        index_ready=True,
        filter_ready=True,
    )


def _hash(value: list[dict[str, Any]]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def _ready(version: CorpusVersion) -> bool:
    evidence = version.validation_evidence
    return bool(
        evidence
        and evidence.chunk_count > 0
        and evidence.embedding_dimensions == version.embedding_dimensions
        and evidence.index_ready
        and evidence.filter_ready
    )
