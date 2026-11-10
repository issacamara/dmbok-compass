"""Persist one immutable pair of administrator-supplied staging artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
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
    StructureTemplate,
    StructureMap,
    StructureMapEntry,
)
from app.corpus.structure_map import HeadingAnchor, SourceIdentity, propose_structure_map
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
    def dispatch(self, *, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any], approval_artifact: dict[str, Any]) -> None: ...


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
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str | None
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
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str | None
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
                if not _valid_artifact(source, bucket_name, attempt_id, "source") or not _valid_artifact(
                    structure, bucket_name, attempt_id, "structure"
                ):
                    raise StagingAttemptConflict("staging attempt artifacts are unavailable")
                if status.preview is None or status.correction_codes or any(
                    entry.match_status != "matched" or entry.correction_codes for entry in status.preview
                ):
                    raise StagingAttemptConflict("staging preview is not fully matched")
                if current.get("preview_binding", {}).get("preview_revision") != status.revision:
                    raise StagingAttemptConflict("staging preview is stale")
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
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> CorpusStagingAttemptStatus:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            record = snapshot.to_dict()
            status = _attempt_status(record)
            if (
                record.get("source") != source
                or record.get("approved_map_artifact") != map_artifact
                or record.get("approval_artifact") != approval_artifact
            ):
                raise StagingAttemptConflict("ingestion artifacts do not match the approved attempt")
            persisted_version_id = record.get("staged_corpus_version_id")
            if (
                corpus_version_id is not None
                and persisted_version_id is not None
                and corpus_version_id != persisted_version_id
            ):
                raise StagingAttemptConflict("staging attempt derived a different corpus version")
            # Duplicate Cloud Run delivery must preserve an already staged result.
            if status.state == "staged":
                return status
            retrying_same_execution = (
                status.state == "failed"
                and state == "running"
                and bool(record.get("dispatch_retryable"))
                and not bool(record.get("dispatch_claimed"))
            )
            if status.state not in {"queued", "running"} and not retrying_same_execution:
                if status.state == state and state in {"staged", "failed"}:
                    return status
                raise StagingAttemptConflict("staging attempt is not available for ingestion progress")
            if state == "staged" and (not corpus_version_id or chunk_count is None):
                raise ValueError("completed staging requires a corpus version and chunk count")
            revision = status.revision + 1
            payload: dict[str, Any] = {
                "state": state, "revision": revision, "completed_stage": "ingestion",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if parent_count is not None:
                payload["parent_count"] = max(parent_count, int(record.get("parent_count") or 0))
            if child_count is not None:
                payload["child_count"] = max(child_count, int(record.get("child_count") or 0))
            if chunk_count is not None:
                payload["chunk_count"] = chunk_count
            if corpus_version_id is not None:
                payload["staged_corpus_version_id"] = corpus_version_id
            if state == "failed":
                payload.update({
                    "correction_codes": [correction_code],
                    "dispatch_retryable": True,
                    "dispatch_claimed": False,
                    "error": ApiError(
                        code=correction_code,
                        message="The approved corpus could not be staged. Retry the ingestion attempt or submit a new attempt.",
                        retryable=True,
                        status=503,
                    ).model_dump(mode="json"),
                })
            else:
                payload["error"] = None
                if retrying_same_execution:
                    payload.update({
                        "dispatch_retryable": False,
                        "dispatch_claimed": True,
                        "correction_codes": [],
                    })
            transaction.update(reference, payload)
            return CorpusStagingAttemptStatus.model_validate({
                **{key: value for key, value in record.items() if key in CorpusStagingAttemptStatus.model_fields},
                **{key: value for key, value in payload.items() if key in CorpusStagingAttemptStatus.model_fields},
                "attempt_id": attempt_id,
            })

        return self._transaction_runner(self.client.transaction(), apply)


def _attempt_status(record: dict[str, Any]) -> CorpusStagingAttemptStatus:
    try:
        status = CorpusStagingAttemptStatus.model_validate(
            {
                key: record[key]
                for key in (
                    "attempt_id",
                    "revision",
                    "state",
                    "completed_stage",
                    "preview",
                    "correction_codes",
                    "parent_count",
                    "child_count",
                    "chunk_count",
                    "staged_corpus_version_id",
                    "parsing_output_sha256",
                    "error",
                )
                if key in record
            }
            | ({"hosted_parsing_state": record["parsing_state"]} if "parsing_state" in record else {})
        )
    except Exception:
        raise StagingPersistenceError("staging attempt is unavailable") from None
    binding = record.get("preview_binding")
    if binding is not None:
        source = record.get("source")
        structure = record.get("structure")
        if (
            not isinstance(binding, dict)
            or not isinstance(source, dict)
            or not isinstance(structure, dict)
            or not isinstance(binding.get("preview_revision"), int)
            or binding["preview_revision"] < 1
            or binding["preview_revision"] > status.revision
            or binding.get("source_generation") != source.get("generation")
            or binding.get("structure_generation") != structure.get("generation")
        ):
            raise StagingPersistenceError("staging preview binding is unavailable")
    return status


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalised_match(value: str) -> str:
    return _NON_ALNUM.sub("", value.casefold())


def _coverage_errors(template: StructureTemplate, page_count: int) -> list[int]:
    missing: list[int] = []
    next_page = 1
    for entry in sorted(template.entries, key=lambda item: (item.page_start, item.page_end)):
        if entry.page_start != next_page or entry.page_end > page_count:
            missing.extend(range(next_page, max(next_page, entry.page_start)))
            if entry.page_start < next_page or entry.page_end > page_count:
                missing.append(entry.page_start)
        next_page = max(next_page, entry.page_end + 1)
    if next_page <= page_count:
        missing.extend(range(next_page, page_count + 1))
    return sorted(set(page for page in missing if 1 <= page <= page_count))


def _preview_for(
    template: StructureTemplate,
    pages: tuple[Any, ...],
    page_count: int,
    *,
    source: dict[str, Any],
) -> tuple[list[CorpusStagingPreviewEntry], list[str]]:
    correction_by_index: dict[int, set[str]] = {index: set() for index in range(len(template.entries))}
    overall: set[str] = set()

    missing_pages = _coverage_errors(template, page_count)
    if missing_pages:
        overall.add("missing_page_coverage")
        for index, entry in enumerate(template.entries):
            if any(
                entry.page_start <= page <= entry.page_end
                or page == entry.page_end + 1
                or page == entry.page_start - 1
                for page in missing_pages
            ):
                correction_by_index[index].add("missing_page_coverage")

    toc = [
        HeadingAnchor(entry.page_start, entry.section_path, entry.section_path[-1])
        for entry in template.entries
        if 1 <= entry.page_start <= page_count
    ]
    printed: list[HeadingAnchor] = []
    previous: str | None = None
    for page in pages:
        heading = page.section
        if not heading or heading == previous:
            previous = heading
            continue
        matching_paths = [
            entry.section_path
            for entry in template.entries
            if _normalised_match(entry.section_path[-1]) == _normalised_match(heading)
        ]
        for section_path in matching_paths or [(heading,)]:
            printed.append(HeadingAnchor(page.page, section_path, heading))
        previous = heading

    try:
        source_identity = SourceIdentity(
            str(source["object_name"]), int(source["generation"]), str(source["sha256"])
        )
        proposal = propose_structure_map(
            source_identity, page_count=page_count, toc=toc, printed_headings=printed
        )
        for unresolved in proposal.unresolved:
            code = "ambiguous_heading" if unresolved.reason == "ambiguous heading anchors" else "unmatched_passage"
            overall.add(code)
            for index, entry in enumerate(template.entries):
                if entry.page_start <= unresolved.page <= entry.page_end:
                    correction_by_index[index].add(code)
    except ValueError:
        overall.add("unmatched_passage")
        for index in range(len(template.entries)):
            correction_by_index[index].add("unmatched_passage")

    preview: list[CorpusStagingPreviewEntry] = []
    for index, entry in enumerate(template.entries):
        codes = sorted(correction_by_index[index])
        preview.append(
            CorpusStagingPreviewEntry(
                entry_index=index,
                section_path=entry.section_path,
                page_start=entry.page_start,
                page_end=entry.page_end,
                match_status="matched" if not codes else (
                    "ambiguous" if "ambiguous_heading" in codes else "unmatched"
                ),
                correction_codes=codes,
            )
        )
    return preview, sorted(overall)


def validate_staging_files(
    *,
    pdf_filename: str | None,
    pdf_content_type: str | None,
    pdf_bytes: bytes,
    map_filename: str | None,
    map_content_type: str | None,
    map_bytes: bytes,
) -> None:
    if len(pdf_bytes) > MAX_STAGING_PDF_BYTES or len(map_bytes) > MAX_STAGING_MAP_BYTES:
        raise StagingUploadValidationError(
            "staging_file_too_large", "One or more staging files exceed the allowed size.", 413
        )
    if not pdf_filename or not pdf_filename.lower().endswith(".pdf") or _media_type(pdf_content_type) != "application/pdf":
        raise StagingUploadValidationError(
            "invalid_staging_pdf", "Upload one PDF file and one JSON structure file.", 415
        )
    if not map_filename or not map_filename.lower().endswith(".json") or _media_type(map_content_type) != "application/json":
        raise StagingUploadValidationError(
            "invalid_staging_map", "Upload one PDF file and one JSON structure file.", 415
        )
    if not pdf_bytes.startswith(b"%PDF-"):
        raise StagingUploadValidationError(
            "invalid_staging_pdf", "The PDF file signature is invalid.", 422
        )
    try:
        StructureTemplate.model_validate_json(map_bytes)
    except ValueError:
        raise StagingUploadValidationError(
            "invalid_staging_map", "The JSON structure file does not match the staging template.", 422
        ) from None


def validate_staging_pdf(*, pdf_filename: str | None, pdf_content_type: str | None, pdf_bytes: bytes) -> None:
    if len(pdf_bytes) > MAX_STAGING_PDF_BYTES:
        raise StagingUploadValidationError(
            "staging_file_too_large", "The PDF exceeds the allowed size.", 413
        )
    if not pdf_filename or not pdf_filename.lower().endswith(".pdf") or _media_type(pdf_content_type) != "application/pdf":
        raise StagingUploadValidationError(
            "invalid_staging_pdf", "Upload one PDF file.", 415
        )
    if not pdf_bytes.startswith(b"%PDF-"):
        raise StagingUploadValidationError(
            "invalid_staging_pdf", "The PDF file signature is invalid.", 422
        )


class CorpusStagingUploadService:
    def __init__(self, store: CorpusStagingAttemptStore, bucket: Any, bucket_name: str) -> None:
        self.store = store
        self.bucket = bucket
        self.bucket_name = bucket_name

    def submit(self, *, actor_id: str, pdf_bytes: bytes, map_bytes: bytes) -> CorpusStagingAttemptStatus:
        attempt_id = uuid4().hex
        source_name = f"staging/{attempt_id}/source.pdf"
        map_name = f"staging/{attempt_id}/structure.json"
        try:
            revision = self.store.begin(
                attempt_id=attempt_id,
                actor_id=actor_id,
                source_object_name=source_name,
                map_object_name=map_name,
            )
        except Exception:
            raise StagingPersistenceError("staging storage is unavailable") from None

        for kind, name, content, content_type in (
            ("source", source_name, pdf_bytes, "application/pdf"),
            ("structure", map_name, map_bytes, "application/json"),
        ):
            try:
                blob = self.bucket.blob(name)
                blob.upload_from_string(
                    content, content_type=content_type, if_generation_match=0, timeout=60
                )
                generation = int(blob.generation)
                if generation < 1:
                    raise ValueError("invalid artifact generation")
                artifact = StagingArtifact(
                    object_name=name,
                    generation=generation,
                    sha256=sha256(content).hexdigest(),
                    content_type=content_type,
                )
                revision = self.store.attach_artifact(
                    attempt_id=attempt_id,
                    artifact_kind=kind,  # type: ignore[arg-type]
                    artifact=artifact.to_dict(bucket_name=self.bucket_name),
                )
            except Exception:
                error = ApiError(
                    code="staging_storage_failed",
                    message="The paired files could not be staged. Please submit a new attempt.",
                    retryable=True,
                    status=503,
                )
                try:
                    return self.store.fail(attempt_id=attempt_id, error=error)
                except Exception:
                    raise StagingPersistenceError("staging storage is unavailable") from None

        return CorpusStagingAttemptStatus(
            attempt_id=attempt_id, revision=revision, state="validating"
        )

    def submit_pdf(self, *, actor_id: str, pdf_bytes: bytes) -> CorpusStagingAttemptStatus:
        """Create a source-only attempt; parsing and structure generation are later stages."""
        if len(pdf_bytes) > MAX_STAGING_PDF_BYTES or not pdf_bytes.startswith(b"%PDF-"):
            raise StagingUploadValidationError(
                "invalid_staging_pdf", "The PDF file is invalid or exceeds the allowed size.", 422
            )
        source_digest = sha256(pdf_bytes).hexdigest()
        attempt_id = sha256(f"{actor_id}\0{source_digest}".encode("utf-8")).hexdigest()[:32]
        try:
            existing = self.store.get(attempt_id=attempt_id)
            if existing is not None:
                source = existing.get("source")
                if existing.get("actor_id") != actor_id or existing.get("source_object_name") != f"staging/{attempt_id}/source.pdf":
                    raise StagingAttemptConflict("source-only attempt identity conflicts")
                if source is not None:
                    if not isinstance(source, dict) or source.get("sha256") != source_digest:
                        raise StagingAttemptConflict("source-only attempt identity conflicts")
                    return _attempt_status(existing)
        except (StagingAttemptConflict, StagingPersistenceError):
            raise
        except Exception:
            raise StagingPersistenceError("staging storage is unavailable") from None
        source_name = f"staging/{attempt_id}/source.pdf"
        try:
            try:
                revision = self.store.begin(
                    attempt_id=attempt_id,
                    actor_id=actor_id,
                    source_object_name=source_name,
                    map_object_name=None,
                )
            except Exception:
                concurrent = self.store.get(attempt_id=attempt_id)
                if (
                    concurrent is None
                    or concurrent.get("actor_id") != actor_id
                    or concurrent.get("source_object_name") != source_name
                ):
                    raise
                attached = concurrent.get("source")
                if isinstance(attached, dict) and attached.get("sha256") == source_digest:
                    return _attempt_status(concurrent)
                revision = int(concurrent.get("revision", 1))
            blob = self.bucket.blob(source_name)
            try:
                blob.upload_from_string(
                    pdf_bytes, content_type="application/pdf", if_generation_match=0, timeout=60
                )
                generation = int(blob.generation)
            except PreconditionFailed:
                existing_blob = self.bucket.blob(source_name)
                existing_blob.reload(timeout=30)
                generation = int(existing_blob.generation)
                existing_bytes = self.bucket.blob(source_name, generation=generation).download_as_bytes(timeout=60)
                if sha256(existing_bytes).hexdigest() != source_digest:
                    raise StagingAttemptConflict("source-only object content conflicts") from None
            if generation < 1:
                raise ValueError("invalid source generation")
            revision = self.store.attach_artifact(
                attempt_id=attempt_id,
                artifact_kind="source",
                artifact=StagingArtifact(
                    object_name=source_name,
                    generation=generation,
                    sha256=source_digest,
                    content_type="application/pdf",
                ).to_dict(bucket_name=self.bucket_name),
            )
        except Exception:
            try:
                self.store.fail(
                    attempt_id=attempt_id,
                    error=ApiError(
                        code="staging_storage_failed",
                        message="The PDF could not be staged. Please submit a new attempt.",
                        retryable=True,
                        status=503,
                    ),
                )
            except Exception:
                pass
            raise StagingPersistenceError("staging storage is unavailable") from None
        return CorpusStagingAttemptStatus(
            attempt_id=attempt_id, revision=revision, state="validating"
        )

    def status(self, *, attempt_id: str) -> CorpusStagingAttemptStatus:
        try:
            record = self.store.get(attempt_id=attempt_id)
            if record is None:
                raise StagingAttemptNotFound("staging attempt is unavailable")
            return _attempt_status(record)
        except (StagingAttemptNotFound, StagingPersistenceError):
            raise
        except Exception:
            raise StagingPersistenceError("staging storage is unavailable") from None

    def validate(self, *, attempt_id: str) -> CorpusStagingAttemptStatus:
        record = self.store.get(attempt_id=attempt_id)
        if record is None:
            raise StagingAttemptNotFound("staging attempt is unavailable")
        current = _attempt_status(record)
        if current.state != "validating":
            return current

        revision = current.revision
        source = record.get("source")
        structure = record.get("structure")
        if not _valid_artifact(source, self.bucket_name, attempt_id, "source") or not _valid_artifact(
            structure, self.bucket_name, attempt_id, "structure"
        ):
            return self._finish_validation(
                attempt_id=attempt_id,
                revision=revision,
                source=source,
                structure=structure,
                status=_failed_status(
                    attempt_id,
                    "source_pair_mismatch",
                    "The staged files could not be matched to this attempt. Submit a new attempt.",
                    preview=None,
                ),
            )

        try:
            source_bytes = _read_pinned_object(self.bucket, source)
            map_bytes = _read_pinned_object(self.bucket, structure)
        except (NotFound, PreconditionFailed):
            return self._finish_validation(
                attempt_id=attempt_id,
                revision=revision,
                source=source,
                structure=structure,
                status=_failed_status(
                    attempt_id,
                    "source_pair_mismatch",
                    "The pinned files are unavailable. Submit a new attempt.",
                    preview=None,
                ),
            )
        except Exception:
            return self._finish_validation(
                attempt_id=attempt_id,
                revision=revision,
                source=source,
                structure=structure,
                status=_failed_status(
                    attempt_id,
                    "job_failed",
                    "The staged files could not be validated. Submit a new attempt.",
                    status=503,
                    retryable=True,
                    preview=None,
                ),
            )

        if sha256(source_bytes).hexdigest() != source["sha256"] or sha256(map_bytes).hexdigest() != structure["sha256"]:
            return self._finish_validation(
                attempt_id=attempt_id,
                revision=revision,
                source=source,
                structure=structure,
                status=_failed_status(
                    attempt_id,
                    "source_pair_mismatch",
                    "The staged files changed after upload. Submit a new attempt.",
                    preview=None,
                ),
            )

        try:
            template = StructureTemplate.model_validate_json(map_bytes)
        except Exception:
            return self._finish_validation(
                attempt_id=attempt_id,
                revision=revision,
                source=source,
                structure=structure,
                status=_failed_status(
                    attempt_id,
                    "invalid_structure",
                    "The structure file does not match the staging template. Submit a new attempt.",
                    preview=None,
                ),
            )

        try:
            with tempfile.TemporaryDirectory(prefix="corpus-staging-") as directory:
                source_path = Path(directory) / "source.pdf"
                source_path.write_bytes(source_bytes)
                report = extract_pdf(source_path, source_uri="staged-source")
                if not _document_title_matches(source_path, template.document_title, report.pages[0].text if report.pages else ""):
                    return self._finish_validation(
                        attempt_id=attempt_id,
                        revision=revision,
                        source=source,
                        structure=structure,
                        status=_failed_status(
                            attempt_id,
                            "source_pair_mismatch",
                            "The structure file does not match the uploaded document. Submit a new attempt.",
                            preview=None,
                        ),
                    )
                if report.source_sha256 != source["sha256"]:
                    return self._finish_validation(
                        attempt_id=attempt_id,
                        revision=revision,
                        source=source,
                        structure=structure,
                        status=_failed_status(
                            attempt_id,
                            "source_pair_mismatch",
                            "The staged PDF does not match its immutable upload. Submit a new attempt.",
                            preview=None,
                        ),
                    )
                preview, corrections = _preview_for(
                    template, report.pages, report.page_count, source=source
                )
                if report.page_count < 1 or len(report.pages) != report.page_count or tuple(
                    page.page for page in report.pages
                ) != tuple(range(1, report.page_count + 1)):
                    corrections = sorted(set(corrections) | {"missing_page_coverage"})
                if report.failures:
                    corrections = sorted(set(corrections) | {"unmatched_passage"})
                    for index, entry in enumerate(preview):
                        if any(entry.page_start <= page <= entry.page_end for page in _failure_pages(report.failures)):
                            codes = sorted(set(entry.correction_codes) | {"unmatched_passage"})
                            preview[index] = entry.model_copy(
                                update={"match_status": "unmatched", "correction_codes": codes}
                            )
                status = (
                    CorpusStagingAttemptStatus(
                        attempt_id=attempt_id,
                        revision=revision,
                        state="failed",
                        completed_stage="validation",
                        preview=preview,
                        correction_codes=corrections,
                        error=ApiError(
                            code="staging_validation_failed",
                            message="The structure map needs correction. Submit a new attempt.",
                            status=422,
                        ),
                    )
                    if corrections
                    else CorpusStagingAttemptStatus(
                        attempt_id=attempt_id,
                        revision=revision,
                        state="awaiting_review",
                        completed_stage="review",
                        preview=preview,
                    )
                )
        except StagingAttemptConflict:
            raise
        except Exception:
            status = _failed_status(
                attempt_id,
                "unmatched_passage",
                "The uploaded document could not be validated. Submit a new attempt.",
                status=422,
                preview=None,
            )

        return self._finish_validation(
            attempt_id=attempt_id,
            revision=revision,
            source=source,
            structure=structure,
            status=status,
        )

    def _finish_validation(
        self,
        *,
        attempt_id: str,
        revision: int,
        source: dict[str, Any] | None,
        structure: dict[str, Any] | None,
        status: CorpusStagingAttemptStatus,
    ) -> CorpusStagingAttemptStatus:
        try:
            return self.store.complete_validation(
                attempt_id=attempt_id,
                expected_revision=revision,
                status=status,
                source_generation=int(source["generation"]) if isinstance(source, dict) else 0,
                structure_generation=int(structure["generation"]) if isinstance(structure, dict) else 0,
            )
        except (StagingAttemptNotFound, StagingAttemptConflict):
            raise
        except Exception:
            raise StagingPersistenceError("staging storage is unavailable") from None


class CloudRunCorpusIngestionDispatcher:
    """Dispatch one pinned source/map/approval tuple to the configured job."""

    def __init__(self, *, project: str, location: str, job_name: str, execute: Callable[[dict[str, str]], None] | None = None) -> None:
        self.project = _required_config(project, "project")
        self.location = _required_config(location, "location")
        self.job_name = _required_config(job_name, "job_name")
        self._execute = execute or self._execute_cloud_run

    def dispatch(self, *, attempt_id: str, source: dict[str, Any], map_artifact: dict[str, Any], approval_artifact: dict[str, Any]) -> None:
        environment = {
            "CORPUS_BUCKET_NAME": source["bucket"],
            "CORPUS_OBJECT_NAME": source["object_name"],
            "CORPUS_SOURCE_GENERATION": str(source["generation"]),
            "CORPUS_STRUCTURE_MAP_URI": _pinned_uri(map_artifact),
            "CORPUS_STRUCTURE_MAP_SHA256": map_artifact["sha256"],
            "CORPUS_STRUCTURE_MAP_APPROVAL_URI": _pinned_uri(approval_artifact),
            "CORPUS_STRUCTURE_MAP_APPROVAL_SHA256": approval_artifact["sha256"],
            "CORPUS_STAGING_ATTEMPT_ID": attempt_id,
        }
        try:
            self._execute(environment)
        except StagingDispatchError:
            raise
        except Exception:
            # Transport failures can happen after Cloud Run accepted the request.
            raise StagingDispatchError(retryable=False) from None

    def _execute_cloud_run(self, environment: dict[str, str]) -> None:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        response = AuthorizedSession(credentials).post(
            "https://run.googleapis.com/v2/projects/"
            f"{self.project}/locations/{self.location}/jobs/{self.job_name}:run",
            json={"overrides": {"containerOverrides": [{"env": [
                {"name": name, "value": value} for name, value in environment.items()
            ]}]}}, timeout=15,
        )
        if not response.ok:
            # A 4xx response is an explicit rejection. Server errors may be
            # returned after job creation and therefore must not be retried.
            raise StagingDispatchError(retryable=400 <= int(response.status_code) < 500)


class CorpusStagingApprovalService:
    def __init__(self, store: CorpusStagingAttemptStore, bucket: Any, bucket_name: str, dispatcher: CorpusIngestionDispatcher) -> None:
        self.store, self.bucket, self.bucket_name, self.dispatcher = store, bucket, bucket_name, dispatcher

    def approve(self, *, attempt_id: str, expected_revision: int, actor_id: str) -> CorpusStagingAttemptStatus:
        record = self.store.get(attempt_id=attempt_id)
        if record is None:
            raise StagingAttemptNotFound("staging attempt is unavailable")
        current = _attempt_status(record)
        if current.state in {"queued", "running", "staged"}:
            return current
        if current.revision != expected_revision or current.state not in {"awaiting_review", "failed"}:
            raise StagingAttemptConflict("staging attempt changed before approval")
        source, template_artifact = record.get("source"), record.get("structure")
        if not _valid_artifact(source, self.bucket_name, attempt_id, "source") or not _valid_artifact(
            template_artifact, self.bucket_name, attempt_id, "structure"
        ):
            raise StagingAttemptConflict("staging artifact pair is invalid")
        if current.state == "awaiting_review" and (
            current.preview is None or current.correction_codes or any(
                entry.match_status != "matched" or entry.correction_codes for entry in current.preview
            ) or record.get("preview_binding", {}).get("preview_revision") != current.revision
        ):
            raise StagingAttemptConflict("staging preview is not fully matched")
        if current.state == "failed" and not record.get("dispatch_retryable"):
            raise StagingAttemptConflict("staging attempt cannot be retried")
        try:
            if current.state == "failed":
                map_artifact = record.get("approved_map_artifact")
                approval_artifact = record.get("approval_artifact")
                map_bytes = self._read_derived(attempt_id, map_artifact, "approved-structure-map.json")
                approval_bytes = self._read_derived(attempt_id, approval_artifact, "approval.json")
                previous = record.get("approval")
                if not isinstance(previous, dict):
                    raise StagingAttemptConflict("the prior approval is unavailable")
                map_value = json.loads(map_bytes)
                expected_source = {
                    "uri": f"gs://{source['bucket']}/{source['object_name']}#{source['generation']}",
                    "generation": source["generation"], "sha256": source["sha256"],
                }
                if previous.get("source") != expected_source or json.loads(approval_bytes) != previous:
                    raise StagingAttemptConflict("the pinned approval does not match this source")
                if map_value.get("source_sha256") != source["sha256"] or previous.get("map", {}).get("content_hash") != map_value.get("content_hash"):
                    raise StagingAttemptConflict("the pinned map does not match this source")
                approval = {
                    **previous, "_source_binding": source, "_structure_binding": template_artifact,
                    "_map_artifact": map_artifact, "_approval_artifact": approval_artifact,
                }
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
            queued, should_dispatch = self.store.claim_approval(
                attempt_id=attempt_id, expected_revision=expected_revision, approval=approval
            )
            if not should_dispatch:
                return queued
            self.dispatcher.dispatch(
                attempt_id=attempt_id, source=source, map_artifact=map_artifact, approval_artifact=approval_artifact
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
