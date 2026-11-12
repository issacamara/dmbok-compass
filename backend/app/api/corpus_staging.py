"""Admin-only corpus staging template and paired-upload endpoints."""

from functools import lru_cache
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.contracts import (
    ApiError,
    CorpusStagingAttemptStatus,
    CorpusStagingStructureCorrectionRequest,
    UserProfile,
)
from app.corpus.staging import (
    CloudRunCorpusIngestionDispatcher,
    CorpusStagingApprovalService,
    CorpusStagingUploadService,
    FirestoreCorpusStagingAttemptStore,
    MAX_STAGING_MAP_BYTES,
    MAX_STAGING_PDF_BYTES,
    StagingAttemptConflict,
    StagingAttemptNotFound,
    StagingPersistenceError,
    StagingUploadValidationError,
    validate_staging_pdf,
    validate_staging_files,
)
from app.ingestion.hosted_parsing import (
    CloudRunHostedParsingDispatcher,
    FirestoreParsingCheckpointStore,
    ParsingAttemptNotFound,
    ParsingDispatchError,
    ParsingError,
    ParsingOutcomeUnknown,
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


@lru_cache(maxsize=1)
def _configured_staging_approval_service() -> CorpusStagingApprovalService:
    bucket_name = os.environ.get("CORPUS_BUCKET_NAME", "").strip()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
    job_name = os.environ.get("INGESTION_JOB_NAME", "").strip()
    if not bucket_name or not project or not location or not job_name:
        raise RuntimeError("staging dispatch is unavailable")
    try:
        from firebase_admin import firestore
        from google.cloud import storage

        from app.identity import _firebase_app

        store = FirestoreCorpusStagingAttemptStore(firestore.client(app=_firebase_app()))
        bucket = storage.Client(project=project).bucket(bucket_name)
        dispatcher = CloudRunCorpusIngestionDispatcher(project=project, location=location, job_name=job_name)
        return CorpusStagingApprovalService(store, bucket, bucket_name, dispatcher)
    except Exception:
        raise RuntimeError("staging dispatch is unavailable") from None


def corpus_staging_approval_service() -> CorpusStagingApprovalService:
    try:
        return _configured_staging_approval_service()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiError(code="staging_dispatch_unavailable", message="Corpus ingestion dispatch is temporarily unavailable.", retryable=True, status=503).model_dump(mode="json"),
        ) from None


@lru_cache(maxsize=1)
def _configured_hosted_parsing_dispatch() -> tuple[FirestoreParsingCheckpointStore, CloudRunHostedParsingDispatcher]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
    job_name = os.environ.get("PARSER_JOB_NAME", "").strip()
    if not project or not location or not job_name:
        raise RuntimeError("hosted parsing dispatch is unavailable")
    try:
        from firebase_admin import firestore

        from app.identity import _firebase_app

        client = firestore.client(app=_firebase_app())
        return (
            FirestoreParsingCheckpointStore(client),
            CloudRunHostedParsingDispatcher(project=project, location=location, job_name=job_name),
        )
    except Exception:
        raise RuntimeError("hosted parsing dispatch is unavailable") from None


def hosted_parsing_dispatch() -> tuple[FirestoreParsingCheckpointStore, CloudRunHostedParsingDispatcher]:
    try:
        return _configured_hosted_parsing_dispatch()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiError(code="parsing_dispatch_unavailable", message="Hosted parsing is temporarily unavailable.", retryable=True, status=503).model_dump(mode="json"),
        ) from None


@router.post(
    "/attempts/pdf",
    response_model=CorpusStagingAttemptStatus,
    status_code=status.HTTP_201_CREATED,
)
async def submit_pdf_only_staging_attempt(
    source_pdf: UploadFile = File(..., media_type="application/pdf"),
    physical_page_numbers: str | None = Form(default=None),
    admin: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        pdf_bytes = await _read_bounded(source_pdf, MAX_STAGING_PDF_BYTES)
        validate_staging_pdf(
            pdf_filename=source_pdf.filename,
            pdf_content_type=source_pdf.content_type,
            pdf_bytes=pdf_bytes,
        )
        selected_pages = _parse_page_selection(physical_page_numbers)
    except StagingUploadValidationError as error:
        raise HTTPException(
            status_code=error.status,
            detail=ApiError(code=error.code, message=error.message, status=error.status).model_dump(mode="json"),
        ) from None
    try:
        return await run_in_threadpool(
            service.submit_pdf, actor_id=admin.user_id, pdf_bytes=pdf_bytes,
            physical_page_numbers=selected_pages,
        )
    except StagingUploadValidationError as error:
        raise HTTPException(
            status_code=error.status,
            detail=ApiError(code=error.code, message=error.message, status=error.status).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None


def _parse_page_selection(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        pages = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise StagingUploadValidationError(
            "invalid_page_selection", "Choose up to five valid physical page numbers.", 422
        ) from None
    if len(pages) > 5 or any(page < 1 for page in pages) or len(set(pages)) != len(pages):
        raise StagingUploadValidationError(
            "invalid_page_selection", "Choose up to five unique physical page numbers starting at 1.", 422
        )
    return sorted(pages)


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


@router.post("/attempts/{attempt_id}/parse", response_model=CorpusStagingAttemptStatus)
def submit_hosted_parse(
    attempt_id: str,
    _: UserProfile = Depends(get_admin_user),
    parsing: tuple[FirestoreParsingCheckpointStore, CloudRunHostedParsingDispatcher] = Depends(hosted_parsing_dispatch),
    status_service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    store, dispatcher = parsing
    try:
        record = store.enqueue(attempt_id)
        if record.get("parsing_state") != "awaiting_review":
            dispatcher.dispatch()
        return status_service.status(attempt_id=attempt_id)
    except (StagingAttemptNotFound, ParsingAttemptNotFound):
        raise _staging_attempt_not_found() from None
    except ParsingOutcomeUnknown:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(code="parsing_outcome_unresolved", message="A previous parse submission needs sponsor reconciliation.", status=409).model_dump(mode="json"),
        ) from None
    except ParsingDispatchError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ApiError(code="parsing_dispatch_failed", message="Hosted parsing could not be started. Retry the attempt.", retryable=True, status=503).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None
    except ParsingError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(code="staging_attempt_not_parseable", message="This attempt is not available for hosted parsing.", status=409).model_dump(mode="json"),
        ) from None


@router.get("/attempts/{attempt_id}", response_model=CorpusStagingAttemptStatus)
def get_corpus_staging_attempt(
    attempt_id: str,
    _: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        return service.status(attempt_id=attempt_id)
    except StagingAttemptNotFound:
        raise _staging_attempt_not_found() from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None


@router.post("/attempts/{attempt_id}/structure/propose", response_model=CorpusStagingAttemptStatus)
def propose_corpus_staging_structure(
    attempt_id: str,
    admin: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        return service.propose_structure(attempt_id=attempt_id, actor_id=admin.user_id)
    except StagingAttemptNotFound:
        raise _staging_attempt_not_found() from None
    except StagingAttemptConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(code="stale_staging_attempt", message="Saved parsing output changed or is not ready for structure review.", status=409).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None


@router.put("/attempts/{attempt_id}/structure", response_model=CorpusStagingAttemptStatus)
def correct_corpus_staging_structure(
    attempt_id: str,
    request: CorpusStagingStructureCorrectionRequest,
    admin: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        return service.correct_structure(
            attempt_id=attempt_id,
            expected_revision=request.expected_revision,
            actor_id=admin.user_id,
            entries=request.entries,
        )
    except StagingAttemptNotFound:
        raise _staging_attempt_not_found() from None
    except StagingAttemptConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(code="stale_staging_attempt", message="This structure review changed. Refresh before saving corrections.", status=409).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None


@router.post("/attempts/{attempt_id}/validate", response_model=CorpusStagingAttemptStatus)
async def validate_corpus_staging_attempt(
    attempt_id: str,
    _: UserProfile = Depends(get_admin_user),
    service: CorpusStagingUploadService = Depends(corpus_staging_upload_service),
) -> CorpusStagingAttemptStatus:
    try:
        return await run_in_threadpool(service.validate, attempt_id=attempt_id)
    except StagingAttemptNotFound:
        raise _staging_attempt_not_found() from None
    except StagingAttemptConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(
                code="stale_staging_attempt",
                message="This staging attempt changed. Submit a new attempt.",
                status=409,
            ).model_dump(mode="json"),
        ) from None
    except StagingPersistenceError:
        raise _staging_unavailable() from None


@router.post("/attempts/{attempt_id}/approve", response_model=CorpusStagingAttemptStatus)
def approve_corpus_staging_attempt(
    attempt_id: str,
    expected_revision: int = Query(ge=1),
    admin: UserProfile = Depends(get_admin_user),
    service: CorpusStagingApprovalService = Depends(corpus_staging_approval_service),
) -> CorpusStagingAttemptStatus:
    try:
        return service.approve(attempt_id=attempt_id, expected_revision=expected_revision, actor_id=admin.user_id)
    except StagingAttemptNotFound:
        raise _staging_attempt_not_found() from None
    except StagingAttemptConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ApiError(code="stale_staging_attempt", message="This staging attempt changed or is not eligible for approval.", status=409).model_dump(mode="json"),
        ) from None
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


def _staging_attempt_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ApiError(
            code="staging_attempt_not_found",
            message="The staging attempt is unavailable.",
            status=404,
        ).model_dump(mode="json"),
    )
