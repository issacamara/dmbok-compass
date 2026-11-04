"""Persist one immutable pair of administrator-supplied staging artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Protocol
from uuid import uuid4

from app.contracts import ApiError, CorpusStagingAttemptStatus, StructureTemplate

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


class FirestoreCorpusStagingAttemptStore:
    """Firestore records keep private artifact identities out of API responses."""

    collection_id = "corpus_staging_attempts"

    def __init__(self, client: Any) -> None:
        self.attempts = client.collection(self.collection_id)

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


def _media_type(value: str | None) -> str:
    return value.split(";", 1)[0].strip().lower() if value else ""
