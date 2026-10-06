from datetime import datetime, timezone
from functools import lru_cache

from fastapi import Depends, FastAPI, Query, status
from fastapi.responses import JSONResponse

from app.contracts.api import AggregateMetric, QuotaPolicy, QuotaStatus, UserProfile
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

app = FastAPI(title="DMBOK Compass API", version="0.1.0")


@app.exception_handler(IdentityError)
async def identity_error_handler(_, exc: IdentityError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(QuotaExceededError)
async def quota_error_handler(_, exc: QuotaExceededError) -> JSONResponse:
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
