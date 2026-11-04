"""Persist one immutable pair of administrator-supplied staging artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Literal, Protocol, TypeVar
from uuid import uuid4

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import firestore

from app.contracts import (
    ApiError,
    CorpusStagingAttemptStatus,
    CorpusStagingPreviewEntry,
    StructureTemplate,
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
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str
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
        self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str
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
        if record.get("state") != "validating" or record.get(artifact_kind) is not None:
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
                    "error",
                )
                if key in record
            }
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
