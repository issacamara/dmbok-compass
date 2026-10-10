from datetime import datetime, timezone
from functools import lru_cache
import os
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.api.evaluations import evaluation_service, router as evaluations_router
from app.answering import QuestionAnswerer
from app.answering.pipeline import GroundedAnswerPolicy, RetrievedAnswerService
from app.contracts.api import (
    AggregateMetric,
    AnswerResponse,
    QuotaPolicy,
    QuotaStatus,
    QuestionRequest,
    ReleaseDecision,
    UserProfile,
)
from app.identity import (
    ApprovalUpdate,
    FirestoreUserRepository,
    IdentityError,
    IdentityPrincipal,
    RegistrationRequest,
    RegistrationStatus,
    get_admin_user,
    get_approved_user,
    get_current_profile,
    get_user_repository,
    verify_firebase_token,
)
from app.metrics import AggregateMetricRepository, FirestoreAggregateMetricRepository
from app.quota import FirestoreQuotaRepository, QuotaExceededError, QuotaRepository
from app.telemetry import LoggingTelemetryAdapter, TelemetryAdapter, error_event, response_event
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.retrieval.search import FirestorePassageRetriever, create_vertex_query_embedder
from model.fallback import FallbackAdapter
from model.gemini import create_gemini_adapter
from model.openrouter import OpenRouterAdapter
from app.evaluation import (
    EvaluationJobService,
    InMemoryEvaluationDatasetStore,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
)
from app.release import ReleaseDecisionError, ReleaseDecisionRequest, ReleaseDecisionService

app = FastAPI(title="DMBOK Compass API", version="0.1.0")
app.include_router(evaluations_router)


@lru_cache(maxsize=1)
def get_telemetry_adapter() -> LoggingTelemetryAdapter:
    return LoggingTelemetryAdapter()


def telemetry_adapter() -> TelemetryAdapter:
    return get_telemetry_adapter()


@lru_cache(maxsize=1)
def get_evaluation_service() -> EvaluationJobService:
    return EvaluationJobService(
        InMemoryEvaluationDatasetStore(),
        InMemoryEvaluationRunStore(),
        InMemoryEvaluationJobDispatcher(),
    )


def configured_evaluation_service() -> EvaluationJobService:
    return get_evaluation_service()


app.dependency_overrides[evaluation_service] = configured_evaluation_service


@lru_cache(maxsize=1)
def get_release_decision_service() -> ReleaseDecisionService:
    service = configured_evaluation_service()
    from app.release import InMemoryReleaseDecisionStore

    return ReleaseDecisionService(service.runs, InMemoryReleaseDecisionStore())


def release_decision_service() -> ReleaseDecisionService:
    return get_release_decision_service()


@app.exception_handler(IdentityError)
async def identity_error_handler(_, exc: IdentityError) -> JSONResponse:
    telemetry_adapter().emit(error_event(exc))
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(QuotaExceededError)
async def quota_error_handler(_, exc: QuotaExceededError) -> JSONResponse:
    telemetry_adapter().emit(error_event(exc))
    status_value = exc.status
    limit = status_value.user_limit if exc.scope == "user" else status_value.global_limit
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": {
                "code": f"{exc.scope}_daily_quota_exceeded",
                "message": str(exc),
                "limit": limit,
                "resets_at": status_value.resets_at.isoformat(),
            }
        },
        headers={
            "Retry-After": str(
                max(1, int((status_value.resets_at - datetime.now(timezone.utc)).total_seconds()))
            )
        },
    )


@lru_cache(maxsize=1)
def get_quota_repository() -> FirestoreQuotaRepository:
    return FirestoreQuotaRepository()


def quota_repository() -> QuotaRepository:
    return get_quota_repository()


@lru_cache(maxsize=1)
def get_aggregate_metric_repository() -> FirestoreAggregateMetricRepository:
    return FirestoreAggregateMetricRepository()


def aggregate_metric_repository() -> AggregateMetricRepository:
    return get_aggregate_metric_repository()


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report that the API process is ready to receive requests."""
    return {"status": "ok"}


@app.post("/api/registration", response_model=UserProfile, status_code=status.HTTP_200_OK)
def register(
    request: RegistrationRequest,
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> UserProfile:
    """Create a pending profile using the verified Firebase identity claims."""
    return repository.register(principal, request.username)


@app.get("/api/registration/status", response_model=RegistrationStatus)
def registration_status(
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> RegistrationStatus:
    profile = repository.refresh_verified_email(principal)
    if profile is None:
        return RegistrationStatus(registered=False)
    return RegistrationStatus(
        registered=True,
        approval_state=profile.approval_state,
        email_verified=profile.email_verified,
    )


@app.get("/api/me", response_model=UserProfile)
def current_user(
    profile: UserProfile = Depends(get_current_profile),
) -> UserProfile:
    return profile


@app.get("/api/quota", response_model=QuotaStatus)
def current_quota(
    profile: UserProfile = Depends(get_approved_user),
    repository: QuotaRepository = Depends(quota_repository),
) -> QuotaStatus:
    """Return only the authenticated user's current UTC-day quota status."""
    return repository.status(profile.user_id)


def reserve_request_quota(
    profile: UserProfile = Depends(get_approved_user),
    repository: QuotaRepository = Depends(quota_repository),
) -> QuotaStatus:
    """Reserve one request before any billable answering work starts."""
    return repository.reserve(profile.user_id)


def question_answerer() -> QuestionAnswerer:
    """Resolve the live Vertex/Firestore/OpenRouter answer pipeline."""
    configured = getattr(app.state, "question_answerer", None)
    return configured or configured_question_answerer()


@lru_cache(maxsize=1)
def configured_question_answerer() -> RetrievedAnswerService:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1")
    if not project:
        raise HTTPException(status_code=503, detail="The Google Cloud project is not configured.")
    import firebase_admin
    from firebase_admin import firestore

    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app()
    client = firestore.client()
    retriever = FirestorePassageRetriever(client, create_vertex_query_embedder(project=project, location=location))
    openrouter = OpenRouterAdapter.from_environment()
    provider = os.environ.get("ANSWER_PROVIDER", "vertex").strip().lower()
    if provider == "openrouter":
        adapter = openrouter
        model_name = os.environ.get("OPENROUTER_PRIMARY_MODEL", "answer-model")
    elif provider == "vertex":
        vertex = create_gemini_adapter(project=project, location=location, vertexai=True)
        adapter = FallbackAdapter(vertex, openrouter)
        model_name = os.environ.get("VERTEX_PRIMARY_MODEL", "gemini-2.5-flash-lite")
    else:
        raise HTTPException(status_code=503, detail="The answer provider is not configured.")
    policy = GroundedAnswerPolicy(
        adapter,
        model_name=model_name,
        timeout_ms=int(os.environ.get("ANSWER_GENERATION_TIMEOUT_MS", "30000")),
    )
    return RetrievedAnswerService(retriever, FirestoreCorpusVersionStore(client), policy)


@app.post("/api/questions", response_model=AnswerResponse)
async def answer_question(
    request: QuestionRequest,
    quota: QuotaStatus = Depends(reserve_request_quota),
    answerer: QuestionAnswerer = Depends(question_answerer),
    telemetry: TelemetryAdapter = Depends(telemetry_adapter),
) -> AnswerResponse:
    """Reserve quota before invoking the request-scoped grounded answerer."""
    started = perf_counter()
    try:
        response = await answerer.answer(request.question, quota)
    except Exception as exc:
        telemetry.emit(error_event(exc, started))
        raise
    telemetry.emit(response_event(response, started))
    return response


@app.get("/api/admin/users", response_model=list[UserProfile])
def list_users(
    approval_state: str | None = Query(default=None, pattern="^(pending|approved|rejected|deactivated)$"),
    limit: int = Query(default=100, ge=1, le=100),
    _: UserProfile = Depends(get_admin_user),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> list[UserProfile]:
    return repository.list_profiles(approval_state, limit)


@app.patch("/api/admin/users/{user_id}", response_model=UserProfile)
def update_user_approval(
    user_id: str,
    request: ApprovalUpdate,
    _: UserProfile = Depends(get_admin_user),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> UserProfile:
    return repository.set_approval(user_id, request.approval_state)


@app.get("/api/admin/configuration", response_model=QuotaPolicy)
def get_configuration(
    _: UserProfile = Depends(get_admin_user),
    repository: QuotaRepository = Depends(quota_repository),
) -> QuotaPolicy:
    return repository.get_policy()


@app.patch("/api/admin/configuration", response_model=QuotaPolicy)
def update_configuration(
    policy: QuotaPolicy,
    _: UserProfile = Depends(get_admin_user),
    repository: QuotaRepository = Depends(quota_repository),
) -> QuotaPolicy:
    return repository.update_policy(policy)


@app.get("/api/admin/metrics", response_model=list[AggregateMetric], tags=["admin"])
def list_aggregate_metrics(
    _: UserProfile = Depends(get_admin_user),
    repository: AggregateMetricRepository = Depends(aggregate_metric_repository),
) -> list[AggregateMetric]:
    """Return allowlisted aggregate metrics without interaction content."""
    return repository.list_metrics()


@app.post("/api/releases", response_model=ReleaseDecision, status_code=status.HTTP_201_CREATED, tags=["admin"])
def record_release_decision(
    request: ReleaseDecisionRequest,
    _: UserProfile = Depends(get_admin_user),
    service: ReleaseDecisionService = Depends(release_decision_service),
) -> ReleaseDecision:
    try:
        return service.record(request)
    except ReleaseDecisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_release_decision", "message": str(exc)},
        ) from exc


@app.get("/api/releases/{release_id}", response_model=ReleaseDecision, tags=["admin"])
def get_release_decision(
    release_id: str,
    _: UserProfile = Depends(get_admin_user),
    service: ReleaseDecisionService = Depends(release_decision_service),
) -> ReleaseDecision:
    try:
        return service.get(release_id)
    except ReleaseDecisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "release_decision_not_found", "message": str(exc)},
        ) from exc
