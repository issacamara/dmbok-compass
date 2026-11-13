Warning: truncated output (original token count: 20663)
Total output lines: 1732

"""Persist one immutable pair of administrator-supplied staging artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import tempfile
import json
from typing import Any, Callable, Literal, Protocol, TypeVar
from uuid import uuid4

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import firestore

from app.contracts import (
    ApiError,
    CorpusStagingAttemptStatus,
    CorpusStagingPreviewEntry,
    CorpusStagingStructureCorrectionEntry,
    StructureTemplate,
    StructureMap,
    StructureMapEntry,
)
from app.contracts.v29 import ParsingArtifact
from app.corpus.structure_map import (
    HeadingAnchor,
    SourceIdentity,
    correct_block_structure_map,
    propose_block_structure_map,
    propose_structure_map,
    validate_block_structure_map,
)
from app.ingestion.extract import extract_pdf

# shortcut: keep fixed upload bounds until representative PDF sizes and API concurrency memory are measured.
MAX_STAGING_PDF_BYTES = 25 * 1024 * 1024
MAX_STAGING_MAP_BYTES = 1024 * 1024


class StagingUploadValidationError(ValueError):
    def __init__(self, code: str, message: str, status: int) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class StagingPersistenceError(RuntimeError):
    """A safe boundary for Firestore or Cloud Storage failures."""


class StagingAttemptNotFound(ValueError):
    """An attempt ID is not present in the private staging store."""


class StagingAttemptConflict(ValueError):
    """An attempt changed before its validation result could be committed."""


class StagingDispatchError(RuntimeError):
    """A safe dispatch failure; retryable means Cloud Run definitively rejected it."""

    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable
        super().__init__("Corpus ingestion dispatch failed")


class CorpusIngestionDispatcher(Protocol):
    def dispatch(
        self, *, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any],
        approval_artifact: dict[str, Any], parsing_artifact: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StagingArtifact:
    object_name: str
    generation: int
    sha256: str
    content_type: Literal["application/pdf", "application/json"]

    def to_dict(self, *, bucket_name: str) -> dict[str, Any]:
        return {
            "bucket": bucket_name,
            "object_name": self.object_name,
            "generation": self.generation,
            "sha256": self.sha256,
            "content_type": self.content_type,
        }


class CorpusStagingAttemptStore(Protocol):
    def begin(
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str | None,
        physical_page_numbers: list[int] | None = None,
    ) -> int: ...

    def attach_artifact(
        self, *, attempt_id: str, artifact_kind: Literal["source", "structure"], artifact: dict[str, Any]
    ) -> int: ...

    def fail(self, *, attempt_id: str, error: ApiError) -> CorpusStagingAttemptStatus: ...

    def get(self, *, attempt_id: str) -> dict[str, Any] | None: ...

    def complete_validation(
        self,
        *,
        attempt_id: str,
        expected_revision: int,
        status: CorpusStagingAttemptStatus,
        source_generation: int,
        structure_generation: int,
    ) -> CorpusStagingAttemptStatus: ...

    def save_structure_review(
        self, *, attempt_id: str, expected_revision: int, parsing_artifact: dict[str, Any],
        structure_artifact: dict[str, Any], status: CorpusStagingAttemptStatus,
        reviewed_by: str,
    ) -> CorpusStagingAttemptStatus: ...

    def claim_approval(
        self, *, attempt_id: str, expected_revision: int, approval: dict[str, Any]
    ) -> tuple[CorpusStagingAttemptStatus, bool]: ...

    def fail_dispatch(
        self, *, attempt_id: str, expected_revision: int, retryable: bool
    ) -> CorpusStagingAttemptStatus: ...

    def record_ingestion_progress(
        self, *, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any],
        approval_artifact: dict[str, Any], state: Literal["running", "staged", "failed"],
        parent_count: int | None = None, child_count: int | None = None,
        chunk_count: int | None = None, corpus_version_id: str | None = None,
        correction_code: Literal["job_interrupted", "job_failed"] = "job_failed",
    ) -> CorpusStagingAttemptStatus: ...


T = TypeVar("T")
_TransactionRunner = Callable[[Any, Callable[[Any], T]], T]


def _run_firestore_transaction(transaction: Any, callback: Callable[[Any], T]) -> T:
    @firestore.transactional
    def run(active_transaction: Any) -> T:
        return callback(active_transaction)

    return run(transaction)


class FirestoreCorpusStagingAttemptStore:
    """Firestore records keep private artifact identities out of API responses."""

    collection_id = "corpus_staging_attempts"

    def __init__(
        self,
        client: Any,
        *,
        transaction_runner: _TransactionRunner = _run_firestore_transaction,
    ) -> None:
        self.client = client
        self.attempts = client.collection(self.collection_id)
        self._transaction_runner = transaction_runner

    def begin(
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str | None,
        physical_page_numbers: list[int] | None = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        self.attempts.document(attempt_id).create(
            {
                "attempt_id": attempt_id,
                "actor_id": actor_id,
                "state": "validating",
                "revision": 1,
                "source_object_name": source_object_name,
                "map_object_name": map_object_name,
                **({"physical_page_numbers": physical_page_numbers} if physical_page_numbers is not None else {}),
                "source": None,
                "structure": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return 1

    def attach_artifact(
        self, *, attempt_id: str, artifact_kind: Literal["source", "structure"], artifact: dict[str, Any]
    ) -> int:
        reference = self.attempts.document(attempt_id)
        snapshot = reference.get()
        if not snapshot.exists:
            raise ValueError("staging attempt is unavailable")
        record = snapshot.to_dict()
        existing = record.get(artifact_kind)
        if existing == artifact:
            return int(record.get("revision", 0))
        if record.get("state") != "validating" or existing is not None:
            raise ValueError("staging attempt cannot accept this artifact")
        revision = int(record.get("revision", 0)) + 1
        reference.update(
            {
                artifact_kind: artifact,
                "revision": revision,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return revision

    def fail(self, *, attempt_id: str, error: ApiError) -> CorpusStagingAttemptStatus:
        reference = self.attempts.document(attempt_id)
        snapshot = reference.get()
        if not snapshot.exists:
            raise ValueError("staging attempt is unavailable")
        revision = int(snapshot.to_dict().get("revision", 0)) + 1
        reference.update(
            {
                "state": "failed",
                "error": error.model_dump(mode="json"),
                "revision": revision,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return CorpusStagingAttemptStatus(
            attempt_id=attempt_id, revision=revision, state="failed", error=error
        )

    def get(self, *, attempt_id: str) -> dict[str, Any] | None:
        snapshot = self.attempts.document(attempt_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def update_candidate_generation_status(
        self,
        *,
        attempt_id: str,
        state: Literal["queued", "running", "awaiting_budget", "published", "failed"],
        completed_count: int | None = None,
        target_count: int | None = None,
        rejected_count: int | None = None,
        usage_tokens: int | None = None,
    ) -> None:
        """Persist only safe candidate-generation progress on the staging attempt."""
        for value in (completed_count, target_count, rejected_count, usage_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("candidate generation progress values must be non-negative integers")
        if (completed_count is None) != (target_count is None):
            raise ValueError("candidate generation progress requires both counts")
        if completed_count is not None and target_count is not None and completed_count > target_count:
            raise ValueError("candidate generation progress exceeds its target")
        payload: dict[str, Any] = {"candidate_generation_state": state}
        if completed_count is not None:
            payload["candidate_generation_count"] = completed_count
            payload["candidate_generation_target"] = target_count
        if rejected_count is not None:
            payload["candidate_generation_rejected_count"] = rejected_count
        if usage_tokens is not None:
            payload["candidate_generation_usage_tokens"] = usage_tokens
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.attempts.document(attempt_id).update(payload)

    def complete_validation(
        self,
        *,
        attempt_id: str,
        expected_revision: int,
        status: CorpusStagingAttemptStatus,
        source_generation: int,
        structure_generation: int,
    ) -> CorpusStagingAttemptStatus:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> CorpusStagingAttemptStatus:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            current = snapshot.to_dict()
            current_status = _attempt_status(current)
            if current_status.state in {"awaiting_review", "failed", "queued", "running", "staged"}:
                return current_status
            if current_status.state != "validating" or current_status.revision != expected_revision:
                raise StagingAttemptConflict("staging attempt changed during validation")

            revision = expected_revision + 1
            result = status.model_copy(update={"revision": revision})
            payload = {
                **result.model_dump(mode="json"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if result.preview is not None:
                payload["preview_binding"] = {
                    "preview_revision": revision,
                    "source_generation": source_generation,
                    "structure_generation": structure_generation,
                }
            transaction.update(reference, payload)
            return result

        return self._transaction_runner(self.client.transaction(), apply)

    def save_structure_review(
        self, *, attempt_id: str, expected_revision: int, parsing_artifact: dict[str, Any],
        structure_artifact: dict[str, Any], status: CorpusStagingAttemptStatus,
        reviewed_by: str,
    ) -> CorpusStagingAttemptStatus:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> CorpusStagingAttemptStatus:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            record = snapshot.to_dict()
            current = _attempt_status(record)
            if current.revision != expected_revision:
                raise StagingAttemptConflict("staging attempt changed before structure review")
            if (
                record.get("parsing_state") != "awaiting_review"
                or record.get("parsing_artifact") != parsing_artifact
                or record.get("approval_artifact") is not None
                or record.get("dispatch_claimed")
                or current.state not in {"validating", "awaiting_review"}
            ):
                raise StagingAttemptConflict("saved parsing output is not available for review")
            source = record.get("source")
            if not isinstance(source, dict) or source.get("generation") is None:
                raise StagingAttemptConflict("staged source is unavailable")
            revision = expected_revision + 1
            result = status.model_copy(update={"revision": revision})
            reviewed_at = datetime.now(timezone.utc).isoformat()
            payload = result.model_dump(mode="json", exclude={"hosted_parsing_state"})
            payload.update({
                "structure": structure_artifact,
                "structure_review": {
                    "revision": revision,
                    "parsing_output_sha256": parsing_artifact.get("sha256"),
                    "map_sha256": structure_artifact["sha256"],
                    "reviewed_by": reviewed_by,
                    "reviewed_at": reviewed_at,
                },
                "preview_binding": {
                    "preview_revision": revision,
                    "source_generation": source["generation"],
                    "structure_generation": structure_artifact["generation"],
                    "parsing_output_sha256": parsing_artifact.get("sha256"),
                },
                "updated_at": reviewed_at,
            })
            transaction.update(reference, payload)
            return result

        return self._transaction_runner(self.client.transaction(), apply)

    def claim_approval(
        self, *, attempt_id: str, expected_revision: int, approval: dict[str, Any]
    ) -> tuple[CorpusStagingAttemptStatus, bool]:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> tuple[CorpusStagingAttemptStatus, bool]:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            current = snapshot.to_dict()
            status = _attempt_status(current)
            if status.state in {"queued", "running", "staged"}:
                return status, False
            retry = status.state == "failed" and bool(current.get("dispatch_retryable"))
            if status.state != "awaiting_review" and not retry:
                raise StagingAttemptConflict("staging attempt is not awaiting approval")
            if status.revision != expected_revision:
                raise StagingAttemptConflict("staging attempt changed before approval")
            if retry and current.get("dispatch_claimed"):
                raise StagingAttemptConflict("staging attempt dispatch is ambiguous")
            if not retry:
                source, structure = current.get("source"), current.get("structure")
                bucket_name = source.get("bucket") if isinstance(source, dict) else ""
                if not _valid_artifact(source, bucket_name, attempt_id, "source") or not _valid_structure_artifact(
                    structure, bucket_name, attempt_id
                ):
                    raise StagingAttemptConflict("staging attempt artifacts are unavailable")
                if status.preview is None or status.correction_codes or any(
                    entry.match_status != "matched" or entry.correction_codes for entry in status.preview
                ):
                    raise StagingAttemptConflict("staging preview is not fully matched")
                if current.get("preview_binding", {}).get("preview_revision") != status.revision:
                    raise StagingAttemptConflict("staging preview is stale")
                parsing = current.get("parsing_artifact")
                if parsing is not None and (
                    not _valid_parsing_identity(parsing, attempt_id, source)
                    or current.get("preview_binding", {}).get("parsing_output_sha256") != parsing.get("sha256")
                    or approval.get("_parsing_binding") != parsing
                ):
                    raise StagingAttemptConflict("staging preview is not bound to the saved parse output")
                if current.get("source") != approval.get("_source_binding") or current.get("structure") != approval.get("_structure_binding"):
                    raise StagingAttemptConflict("staging artifact pair changed")
            revision = status.revision + 1
            payload = {
                "state": "queued", "revision": revision, "completed_stage": "approval",
                "approval": {key: value for key, value in approval.items() if not key.startswith("_")},
                "approved_map_artifact": approval.get("_map_artifact"),
                "approval_artifact": approval.get("_approval_artifact"),
                "dispatch_claimed": True, "dispatch_retryable": False,
                "error": None, "correction_codes": [],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            transaction.update(reference, payload)
            return CorpusStagingAttemptStatus(attempt_id=attempt_id, revision=revision, state="queued", completed_stage="approval"), True

        return self._transaction_runner(self.client.transaction(), apply)

    def fail_dispatch(
        self, *, attempt_id: str, expected_revision: int, retryable: bool
    ) -> CorpusStagingAttemptStatus:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> CorpusStagingAttemptStatus:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            current = snapshot.to_dict()
            status = _attempt_status(current)
            if status.state != "queued" or status.revision != expected_revision:
                return status
            revision = status.revision + 1
            error = ApiError(
                code="job_failed", message="The ingestion job could not be dispatched. Retry approval or submit a new attempt.",
                retryable=retryable, status=503,
            )
            transaction.update(reference, {
                "state": "failed", "revision": revision, "error": error.model_dump(mode="json"),
                "dispatch_retryable": retryable,
                # Keep the claim on ambiguous outcomes; the request may have been accepted.
                "dispatch_claimed": not retryable, "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            return CorpusStagingAttemptStatus(attempt_id=attempt_id, revision=revision, state="failed", completed_stage="approval", error=error)

        return self._transaction_runner(self.client.transaction(), apply)

    def record_ingestion_progress(
        self, *, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any],
        approval_artifact: dict[str, Any], state: Literal["running", "staged", "failed"],
        parent_count: int | None = None, child_count: int | None = None,
        chunk_count: int | None = None, corpus_version_id: str | None = None,
        correction_code: Literal["job_interrupted", "job_failed"] = "job_failed",
    ) -> CorpusStagingAttemptStatus:
        """Persist safe progress only for the exact approved artifact tuple."""
        …10663 tokens truncated…": source["sha256"],
                }
                if previous.get("source") != expected_source or json.loads(approval_bytes) != previous:
                    raise StagingAttemptConflict("the pinned approval does not match this source")
                if map_value.get("source_sha256") != source["sha256"] or previous.get("map", {}).get("content_hash") != map_value.get("content_hash"):
                    raise StagingAttemptConflict("the pinned map does not match this source")
                if parse_bound:
                    parsing = self._read_parsing_artifact(attempt_id, source, parsing_identity)
                    structure_map = StructureMap.model_validate_json(map_bytes)
                    validate_block_structure_map(parsing, structure_map)
                    if previous.get("parsing", {}).get("output_sha256") != parsing.output_sha256:
                        raise StagingAttemptConflict("the pinned approval does not match the saved parse output")
                approval = {
                    **previous, "_source_binding": source, "_structure_binding": template_artifact,
                    "_map_artifact": map_artifact, "_approval_artifact": approval_artifact,
                    "_parsing_binding": parsing_identity,
                }
                parsing_dispatch = self._parsing_dispatch_identity(source, parsing_identity) if parse_bound else None
            else:
                if parse_bound:
                    parsing = self._read_parsing_artifact(attempt_id, source, parsing_identity)
                    map_bytes = self._read_reviewed_map(attempt_id, template_artifact)
                    structure_map = StructureMap.model_validate_json(map_bytes)
                    validate_block_structure_map(parsing, structure_map)
                    map_artifact = template_artifact
                    _, approval_artifact = self._persist_parse_bound_approval(
                        attempt_id, source, parsing_identity, map_artifact, map_bytes,
                        actor_id=actor_id, review_revision=current.revision,
                    )
                    map_value = structure_map.model_dump(mode="json")
                    approval = {
                        "schema_version": "corpus-owner-approval-v1", "decision": "approved",
                        "approved_by": actor_id,
                        "source": {
                            "uri": f"gs://{source['bucket']}/{source['object_name']}#{source['generation']}",
                            "generation": source["generation"], "sha256": source["sha256"],
                        },
                        "parsing": {
                            "artifact_uri": f"gs://{source['bucket']}/{parsing_identity['object_name']}#{parsing_identity['artifact_generation']}",
                            "artifact_sha256": parsing_identity["artifact_sha256"],
                            "output_sha256": parsing_identity["sha256"],
                        },
                        "map": {
                            "artifact_uri": f"gs://{map_artifact['bucket']}/{map_artifact['object_name']}#{map_artifact['generation']}",
                            "generation": map_artifact["generation"], "file_sha256": map_artifact["sha256"],
                            "content_hash": map_value["content_hash"], "map_id": map_value["map_id"],
                            "schema_version": "structure-map-v1",
                        },
                        "review_revision": current.revision,
                        "ingestion_gate": {"oversized_indivisible_list_items_or_table_rows": True},
                        "_source_binding": source, "_structure_binding": template_artifact,
                        "_parsing_binding": parsing_identity,
                        "_map_artifact": map_artifact, "_approval_artifact": approval_artifact,
                    }
                    parsing_dispatch = self._parsing_dispatch_identity(source, parsing_identity)
                else:
                    map_bytes, map_artifact = self._persist_map(attempt_id, source, template_artifact)
                    _, approval_artifact = self._persist_approval(
                        attempt_id, source, map_artifact, map_bytes, actor_id=actor_id
                    )
                    map_value = json.loads(map_bytes)
                    approval = {
                        "schema_version": "corpus-owner-approval-v1", "decision": "approved", "approved_by": actor_id,
                        "source": {
                            "uri": f"gs://{source['bucket']}/{source['object_name']}#{source['generation']}",
                            "generation": source["generation"], "sha256": source["sha256"],
                        },
                        "map": {
                            "artifact_uri": f"gs://{map_artifact['bucket']}/{map_artifact['object_name']}",
                            "generation": map_artifact["generation"], "file_sha256": map_artifact["sha256"],
                            "content_hash": map_value["content_hash"], "map_id": map_value["map_id"],
                            "schema_version": "structure-map-v1",
                        },
                        "ingestion_gate": {"oversized_indivisible_list_items_or_table_rows": True},
                        "_source_binding": source, "_structure_binding": template_artifact,
                        "_map_artifact": map_artifact, "_approval_artifact": approval_artifact,
                    }
                    parsing_dispatch = None
            queued, should_dispatch = self.store.claim_approval(
                attempt_id=attempt_id, expected_revision=expected_revision, approval=approval
            )
            if not should_dispatch:
                return queued
            self.dispatcher.dispatch(
                attempt_id=attempt_id, source=source, map_artifact=map_artifact,
                approval_artifact=approval_artifact, parsing_artifact=parsing_dispatch,
            )
            return queued
        except StagingDispatchError as error:
            try:
                return self.store.fail_dispatch(
                    attempt_id=attempt_id, expected_revision=current.revision + 1, retryable=error.retryable
                )
            except Exception:
                raise StagingPersistenceError("staging dispatch status is unavailable") from None
        except (StagingAttemptNotFound, StagingAttemptConflict):
            raise
        except Exception:
            raise StagingPersistenceError("staging approval is unavailable") from None

    def _read_parsing_artifact(
        self, attempt_id: str, source: dict[str, Any], identity: dict[str, Any]
    ) -> ParsingArtifact:
        if not _valid_parsing_identity(identity, attempt_id, source):
            raise StagingAttemptConflict("the saved parsing artifact is unavailable")
        try:
            payload = self.bucket.blob(
                identity["object_name"], generation=identity["artifact_generation"]
            ).download_as_bytes(timeout=60)
            if sha256(payload).hexdigest() != identity["artifact_sha256"]:
                raise ValueError("parsing artifact digest mismatch")
            parsing = ParsingArtifact.model_validate_json(payload)
            if parsing.output_sha256 != identity["sha256"]:
                raise ValueError("parsing output digest mismatch")
            return parsing
        except Exception:
            raise StagingAttemptConflict("the saved parsing artifact is invalid") from None

    def _parsing_dispatch_identity(
        self, source: dict[str, Any], identity: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "bucket": source["bucket"],
            "object_name": identity["object_name"],
            "generation": identity["artifact_generation"],
            "sha256": identity["artifact_sha256"],
            "output_sha256": identity["sha256"],
        }

    def _read_reviewed_map(self, attempt_id: str, artifact: Any) -> bytes:
        if not _valid_structure_artifact(artifact, self.bucket_name, attempt_id):
            raise StagingAttemptConflict("the reviewed structure map is unavailable")
        try:
            payload = _read_pinned_object(self.bucket, artifact)
            if sha256(payload).hexdigest() != artifact["sha256"]:
                raise ValueError("structure map digest mismatch")
            return payload
        except Exception:
            raise StagingAttemptConflict("the reviewed structure map is invalid") from None

    def _persist_parse_bound_approval(
        self, attempt_id: str, source: dict[str, Any], parsing_identity: dict[str, Any],
        map_artifact: dict[str, Any], map_bytes: bytes, *, actor_id: str, review_revision: int,
    ) -> tuple[bytes, dict[str, Any]]:
        structure_map = StructureMap.model_validate_json(map_bytes)
        value = {
            "schema_version": "corpus-owner-approval-v1",
            "decision": "approved",
            "approved_by": actor_id,
            "source": {
                "uri": f"gs://{source['bucket']}/{source['object_name']}#{source['generation']}",
                "generation": source["generation"], "sha256": source["sha256"],
            },
            "parsing": {
                "artifact_uri": f"gs://{source['bucket']}/{parsing_identity['object_name']}#{parsing_identity['artifact_generation']}",
                "artifact_sha256": parsing_identity["artifact_sha256"],
                "output_sha256": parsing_identity["sha256"],
            },
            "map": {
                "artifact_uri": f"gs://{map_artifact['bucket']}/{map_artifact['object_name']}#{map_artifact['generation']}",
                "generation": map_artifact["generation"], "file_sha256": map_artifact["sha256"],
                "content_hash": structure_map.content_hash, "map_id": structure_map.map_id,
                "schema_version": structure_map.schema_version,
            },
            "review_revision": review_revision,
            "ingestion_gate": {"oversized_indivisible_list_items_or_table_rows": True},
        }
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return payload, self._write_pinned(f"staging/{attempt_id}/approval.json", payload, "application/json")

    def _persist_map(self, attempt_id: str, source: dict[str, Any], artifact: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        template_bytes = _read_pinned_object(self.bucket, artifact)
        if sha256(template_bytes).hexdigest() != artifact["sha256"]:
            raise StagingAttemptConflict("staging structure changed")
        template = StructureTemplate.model_validate_json(template_bytes)
        material = [
            {"section_path": item.section_path, "page_start": item.page_start, "page_end": item.page_end}
            for item in template.entries
        ]
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        content_hash = sha256(encoded).hexdigest()
        structure_map = StructureMap(
            map_id=f"staged-{source['sha256'][:12]}-{content_hash[:12]}",
            source_sha256=source["sha256"], content_hash=content_hash,
            entries=[StructureMapEntry.model_validate(item) for item in material],
        )
        payload = (json.dumps(structure_map.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n").encode()
        return payload, self._write_pinned(f"staging/{attempt_id}/approved-structure-map.json", payload, "application/json")

    def _persist_approval(self, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any], map_bytes: bytes, *, actor_id: str) -> tuple[bytes, dict[str, Any]]:
        structure_map = json.loads(map_bytes)
        value = {
            "schema_version": "corpus-owner-approval-v1", "decision": "approved", "approved_by": actor_id,
            "source": {"uri": f"gs://{source['bucket']}/{source['object_name']}#{source['generation']}", "generation": source["generation"], "sha256": source["sha256"]},
            "map": {"artifact_uri": f"gs://{map_artifact['bucket']}/{map_artifact['object_name']}", "generation": map_artifact["generation"], "file_sha256": map_artifact["sha256"], "content_hash": structure_map["content_hash"], "map_id": structure_map["map_id"], "schema_version": structure_map["schema_version"]},
            "ingestion_gate": {"oversized_indivisible_list_items_or_table_rows": True},
        }
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return payload, self._write_pinned(f"staging/{attempt_id}/approval.json", payload, "application/json")

    def _read_derived(self, attempt_id: str, artifact: Any, suffix: str) -> bytes:
        if not isinstance(artifact, dict) or not _valid_derived_artifact(
            artifact, self.bucket_name, f"staging/{attempt_id}/{suffix}"
        ):
            raise StagingAttemptConflict("the pinned approval artifacts are unavailable")
        payload = _read_pinned_object(self.bucket, artifact)
        if sha256(payload).hexdigest() != artifact["sha256"]:
            raise StagingAttemptConflict("the pinned approval artifacts changed")
        return payload

    def _write_pinned(self, object_name: str, payload: bytes, content_type: str) -> dict[str, Any]:
        blob = self.bucket.blob(object_name)
        try:
            blob.upload_from_string(payload, content_type=content_type, if_generation_match=0, timeout=60)
            generation = int(blob.generation)
        except Exception:
            # Idempotent duplicate clicks reuse only the exact already-pinned bytes.
            existing = self.bucket.blob(object_name)
            existing.reload()
            generation = int(existing.generation)
            pinned = self.bucket.blob(object_name, generation=generation).download_as_bytes(timeout=60)
            if pinned != payload:
                raise StagingAttemptConflict("pinned approval artifact does not match this attempt") from None
        return {"bucket": self.bucket_name, "object_name": object_name, "generation": generation,
                "sha256": sha256(payload).hexdigest(), "content_type": content_type}


def _pinned_uri(artifact: dict[str, Any]) -> str:
    return f"gs://{artifact['bucket']}/{artifact['object_name']}#{artifact['generation']}"


def _valid_derived_artifact(artifact: dict[str, Any], bucket_name: str, object_name: str) -> bool:
    return (
        artifact.get("bucket") == bucket_name
        and artifact.get("object_name") == object_name
        and isinstance(artifact.get("generation"), int) and artifact["generation"] > 0
        and isinstance(artifact.get("sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]) is not None
        and artifact.get("content_type") == "application/json"
    )


def _required_config(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _valid_artifact(
    artifact: Any, bucket_name: str, attempt_id: str, kind: Literal["source", "structure"]
) -> bool:
    if not isinstance(artifact, dict):
        return False
    object_name = f"staging/{attempt_id}/{'source.pdf' if kind == 'source' else 'structure.json'}"
    return (
        artifact.get("bucket") == bucket_name
        and artifact.get("object_name") == object_name
        and isinstance(artifact.get("generation"), int)
        and artifact["generation"] > 0
        and isinstance(artifact.get("sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]) is not None
        and artifact.get("content_type") == ("application/pdf" if kind == "source" else "application/json")
    )


def _valid_structure_artifact(artifact: Any, bucket_name: str, attempt_id: str) -> bool:
    return _valid_artifact(artifact, bucket_name, attempt_id, "structure") or (
        isinstance(artifact, dict)
        and artifact.get("bucket") == bucket_name
        and re.fullmatch(
            rf"staging/{re.escape(attempt_id)}/structure-revisions/\d+-[a-f0-9]{{64}}\.json",
            str(artifact.get("object_name", "")),
        ) is not None
        and isinstance(artifact.get("generation"), int)
        and artifact["generation"] > 0
        and isinstance(artifact.get("sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", artifact["sha256"]) is not None
        and artifact.get("content_type") == "application/json"
    )


def _valid_parsing_identity(identity: Any, attempt_id: str, source: Any) -> bool:
    return (
        isinstance(identity, dict)
        and isinstance(source, dict)
        and identity.get("source_generation") == str(source.get("generation"))
        and identity.get("source_sha256") == source.get("sha256")
        and identity.get("object_name") == f"staging/{attempt_id}/parsing/{identity.get('sha256')}.json"
        and isinstance(identity.get("sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", identity["sha256"]) is not None
        and isinstance(identity.get("artifact_sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", identity["artifact_sha256"]) is not None
        and isinstance(identity.get("artifact_generation"), int)
        and identity["artifact_generation"] > 0
        and isinstance(identity.get("remote_job_id"), str)
    )


def _write_pinned_artifact(
    bucket: Any, bucket_name: str, object_name: str, payload: bytes, content_type: str
) -> dict[str, Any]:
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_string(payload, content_type=content_type, if_generation_match=0, timeout=60)
        generation = int(blob.generation)
    except PreconditionFailed:
        existing = bucket.blob(object_name)
        existing.reload()
        generation = int(existing.generation)
        pinned = bucket.blob(object_name, generation=generation).download_as_bytes(timeout=60)
        if pinned != payload:
            raise StagingAttemptConflict("immutable structure revision conflicts") from None
    return {
        "bucket": bucket_name,
        "object_name": object_name,
        "generation": generation,
        "sha256": sha256(payload).hexdigest(),
        "content_type": content_type,
    }


def _read_pinned_object(bucket: Any, artifact: dict[str, Any]) -> bytes:
    blob = bucket.blob(artifact["object_name"], generation=artifact["generation"])
    return blob.download_as_bytes(timeout=60)


def _failed_status(
    attempt_id: str,
    correction_code: str,
    message: str,
    *,
    status: int = 422,
    retryable: bool = False,
    preview: list[CorpusStagingPreviewEntry] | None,
) -> CorpusStagingAttemptStatus:
    return CorpusStagingAttemptStatus(
        attempt_id=attempt_id,
        revision=1,
        state="failed",
        completed_stage="validation",
        preview=preview,
        correction_codes=[correction_code],
        error=ApiError(
            code="staging_validation_failed",
            message=message,
            retryable=retryable,
            status=status,
        ),
    )


def _document_title_matches(pdf_path: Path, title: str, first_page: str) -> bool:
    normalized = _normalised_match(title)
    if not normalized:
        return False
    if normalized in _normalised_match(first_page):
        return True
    try:
        from pypdf import PdfReader

        metadata_title = PdfReader(str(pdf_path)).metadata.title or ""
        return normalized in _normalised_match(metadata_title)
    except Exception:
        return False


def _failure_pages(failures: tuple[str, ...]) -> set[int]:
    pages: set[int] = set()
    for failure in failures:
        match = re.match(r"page (\d+):", failure)
        if match:
            pages.add(int(match.group(1)))
    return pages


def _media_type(value: str | None) -> str:
    return value.split(";", 1)[0].strip().lower() if value else ""
