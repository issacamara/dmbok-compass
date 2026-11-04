"""Admin-only corpus staging template and paired-upload endpoints."""

from functools import lru_cache
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.contracts import ApiError, CorpusStagingAttemptStatus, UserProfile
from app.corpus.staging import (
    CorpusStagingUploadService,
    FirestoreCorpusStagingAttemptStore,
    MAX_STAGING_MAP_BYTES,
    MAX_STAGING_PDF_BYTES,
    StagingPersistenceError,
    StagingUploadValidationError,
    validate_staging_files,
)
from app.identity import get_admin_user


router = APIRouter(prefix="/api/admin/corpus-staging", tags=["corpus staging"])
TEMPLATE_PATH = Path(__file__).parents[1] / "templates" / "dmbok-structure-template-v1.json"


@router.get("/template", response_class=FileResponse)
def download_structure_template(
    _: UserProfile = Depends(get_admin_user),
) -> FileResponse:
    return FileResponse(
        TEMPLATE_PATH,
        media_type="application/json",
        filename="dmbok-structure-template-v1.json",
    )


@lru_cache(maxsize=1)
def _configured_staging_upload_service() -> CorpusStagingUploadService:
    bucket_name = os.environ.get("CORPUS_BUCKET_NAME", "").strip()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not bucket_name or not project:
        raise RuntimeError("staging storage is unavailable")
    try:
        from firebase_admin import firestore
        from google.cloud import storage

        from app.identity import _firebase_app

        store = FirestoreCorpusStagingAttemptStore(firestore.client(app=_firebase_app()))
        bucket = storage.Client(project=project).bucket(bucket_name)
        return CorpusStagingUploadService(store, bucket, bucket_name)
    except Exception:
        raise RuntimeError("staging storage is unavailable") from None


def corpus_staging_upload_service() -> CorpusStagingUploadService:
    try:
        return _configured_staging_upload_service()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiError(
                code="staging_storage_unavailable",
                message="Corpus staging is temporarily unavailable.",
                retryable=True,
                status=503,
            ).model_dump(mode="json"),
        ) from None


@router.post(
    "/attempts",
    response_model=CorpusStagingAttemptStatus,
    status_code=status.HTTP_201_CREATED,
)
async def submit_corpus_staging_attempt(
    source_pdf: UploadFile = File(..., media_type="application/pdf"),
    structure_json: UploadFile = File(..., media_type="application/json"),
    admin: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        pdf_bytes = await _read_bounded(source_pdf, MAX_STAGING_PDF_BYTES)
        map_bytes = await _read_bounded(structure_json, MAX_STAGING_MAP_BYTES)
        validate_staging_files(
            pdf_filename=source_pdf.filename,
            pdf_content_type=source_pdf.content_type,
            pdf_bytes=pdf_bytes,
            map_filename=structure_json.filename,
            map_content_type=structure_json.content_type,
            map_bytes=map_bytes,
        )
    except StagingUploadValidationError as error:
        raise HTTPException(
            status_code=error.status,
            detail=ApiError(code=error.code, message=error.message, status=error.status).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None

    try:
        return await run_in_threadpool(
            service.submit, actor_id=admin.user_id, pdf_bytes=pdf_bytes, map_bytes=map_bytes
        )
    except StagingPersistenceError:
        raise _staging_unavailable() from None


async def _read_bounded(file: UploadFile, maximum: int) -> bytes:
    if file.size is not None and file.size > maximum:
        raise StagingUploadValidationError(
            "staging_file_too_large", "One or more staging files exceed the allowed size.", 413
        )
    content = await file.read(maximum + 1)
    if len(content) > maximum:
        raise StagingUploadValidationError(
            "staging_file_too_large", "One or more staging files exceed the allowed size.", 413
        )
    return content


def _staging_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=ApiError(
            code="staging_storage_unavailable",
            message="Corpus staging is temporarily unavailable.",
            retryable=True,
            status=503,
        ).model_dump(mode="json"),
    )
