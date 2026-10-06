from fastapi import Depends, FastAPI, Query, status
from fastapi.responses import JSONResponse

from app.contracts.api import UserProfile
from app.identity import (
    ApprovalUpdate,
    FirestoreUserRepository,
    IdentityError,
    IdentityPrincipal,
    RegistrationRequest,
    RegistrationStatus,
    current_profile,
    get_user_repository,
    require_admin,
    verify_firebase_token,
)

app = FastAPI(title="DMBOK Compass API", version="0.1.0")


@app.exception_handler(IdentityError)
async def identity_error_handler(_, exc: IdentityError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


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
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> UserProfile:
    return current_profile(principal, repository)


@app.get("/api/admin/users", response_model=list[UserProfile])
def list_users(
    approval_state: str | None = Query(default=None, pattern="^(pending|approved|rejected|deactivated)$"),
    limit: int = Query(default=100, ge=1, le=100),
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> list[UserProfile]:
    admin = current_profile(principal, repository)
    require_admin(admin)
    return repository.list_profiles(approval_state, limit)


@app.patch("/api/admin/users/{user_id}", response_model=UserProfile)
def update_user_approval(
    user_id: str,
    request: ApprovalUpdate,
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> UserProfile:
    admin = current_profile(principal, repository)
    require_admin(admin)
    return repository.set_approval(user_id, request.approval_state)
