"""Firebase token verification and Firestore-backed account approval."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import firebase_admin
from fastapi import Depends, Header, HTTPException, status
from firebase_admin import auth, firestore
from google.cloud import firestore as google_firestore
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.api import UserProfile

APPROVED_USER_LIMIT = 10


class IdentityPrincipal(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    email: str = Field(min_length=3, max_length=320)
    email_verified: bool


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    username: str = Field(min_length=1, max_length=80)


class RegistrationStatus(BaseModel):
    registered: bool
    approval_state: Literal["pending", "approved", "rejected", "deactivated"] | None = None
    email_verified: bool | None = None


class ApprovalUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_state: Literal["approved", "rejected", "deactivated"]


class IdentityError(Exception):
    def __init__(self, detail: str, status_code: int = status.HTTP_409_CONFLICT):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def _firebase_app() -> firebase_admin.App:
    try:
        return firebase_admin.get_app()
    except ValueError:
        return firebase_admin.initialize_app()


def verify_firebase_token(authorization: str | None = Header(default=None)) -> IdentityPrincipal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A Firebase ID token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = auth.verify_id_token(token, app=_firebase_app(), check_revoked=True)
        email = claims.get("email")
        user_id = claims.get("uid") or claims.get("sub")
        if not user_id or not email:
            raise ValueError("Required identity claims are missing.")
        return IdentityPrincipal(
            user_id=str(user_id),
            email=str(email),
            email_verified=bool(claims.get("email_verified", False)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase ID token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _next_approved_count(current: int, was_approved: bool, will_be_approved: bool) -> int:
    if will_be_approved and not was_approved:
        if current >= APPROVED_USER_LIMIT:
            raise IdentityError("The approved-user limit has been reached.", status.HTTP_409_CONFLICT)
        return current + 1
    if was_approved and not will_be_approved:
        return max(0, current - 1)
    return current


class FirestoreUserRepository:
    """Server-only access to profiles and the serialized approval counter."""

    def __init__(self) -> None:
        self.client = firestore.client(app=_firebase_app())
        self.users = self.client.collection("users")
        self.capacity = self.client.collection("application_configuration").document("user_access")

    def get(self, user_id: str) -> UserProfile | None:
        snapshot = self.users.document(user_id).get()
        if not snapshot.exists:
            return None
        return UserProfile.model_validate(snapshot.to_dict())

    def register(self, principal: IdentityPrincipal, username: str) -> UserProfile:
        profile_ref = self.users.document(principal.user_id)
        transaction = self.client.transaction()

        @google_firestore.transactional
        def write_profile(transaction):  # type: ignore[no-untyped-def]
            snapshot = profile_ref.get(transaction=transaction)
            if snapshot.exists:
                stored = UserProfile.model_validate(snapshot.to_dict())
                updated = stored.model_copy(
                    update={
                        "email": str(principal.email),
                        "email_verified": stored.email_verified or principal.email_verified,
                    }
                )
                transaction.set(profile_ref, updated.model_dump())
                return updated
            profile = UserProfile(
                user_id=principal.user_id,
                email=str(principal.email),
                username=username,
                approval_state="pending",
                role="user",
                email_verified=principal.email_verified,
            )
            transaction.create(profile_ref, profile.model_dump())
            return profile

        return write_profile(transaction)

    def refresh_verified_email(self, principal: IdentityPrincipal) -> UserProfile | None:
        profile_ref = self.users.document(principal.user_id)
        transaction = self.client.transaction()

        @google_firestore.transactional
        def refresh(transaction):  # type: ignore[no-untyped-def]
            snapshot = profile_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None
            profile = UserProfile.model_validate(snapshot.to_dict())
            updated = profile.model_copy(
                update={
                    "email": str(principal.email),
                    "email_verified": profile.email_verified or principal.email_verified,
                }
            )
            if updated != profile:
                transaction.set(profile_ref, updated.model_dump())
            return updated

        return refresh(transaction)

    def list_profiles(self, approval_state: str | None, limit: int) -> list[UserProfile]:
        query = self.users
        if approval_state:
            query = query.where("approval_state", "==", approval_state)
        return [UserProfile.model_validate(doc.to_dict()) for doc in query.limit(limit).stream()]

    def set_approval(self, user_id: str, approval_state: str) -> UserProfile:
        profile_ref = self.users.document(user_id)
        transaction = self.client.transaction()

        @google_firestore.transactional
        def update_profile(transaction):  # type: ignore[no-untyped-def]
            profile_snapshot = profile_ref.get(transaction=transaction)
            if not profile_snapshot.exists:
                raise IdentityError("User profile was not found.", status.HTTP_404_NOT_FOUND)
            profile = UserProfile.model_validate(profile_snapshot.to_dict())
            if approval_state == "approved" and not profile.email_verified:
                raise IdentityError("The user must verify their email before approval.")

            capacity_snapshot = self.capacity.get(transaction=transaction)
            if capacity_snapshot.exists:
                approved_count = int(capacity_snapshot.get("approved_user_count"))
            else:
                approved_query = self.users.where("approval_state", "==", "approved")
                approved_count = sum(1 for _ in approved_query.stream(transaction=transaction))

            next_count = _next_approved_count(
                approved_count,
                profile.approval_state == "approved",
                approval_state == "approved",
            )
            updated = profile.model_copy(update={"approval_state": approval_state})
            transaction.set(profile_ref, updated.model_dump())
            transaction.set(
                self.capacity,
                {"approved_user_count": next_count, "approved_user_limit": APPROVED_USER_LIMIT},
            )
            return updated

        try:
            return update_profile(transaction)
        except IdentityError:
            raise


@lru_cache(maxsize=1)
def get_user_repository() -> FirestoreUserRepository:
    return FirestoreUserRepository()


def current_profile(
    principal: IdentityPrincipal,
    repository: FirestoreUserRepository,
) -> UserProfile:
    profile = repository.refresh_verified_email(principal)
    if profile is None:
        raise IdentityError("Registration has not been completed.", status.HTTP_404_NOT_FOUND)
    return profile


def require_approved_user(profile: UserProfile) -> UserProfile:
    if not profile.email_verified:
        raise IdentityError("Verify your email before using DMBOK Compass.", status.HTTP_403_FORBIDDEN)
    if profile.approval_state != "approved":
        raise IdentityError("Your account is not approved for access.", status.HTTP_403_FORBIDDEN)
    return profile


def require_admin(profile: UserProfile) -> UserProfile:
    require_approved_user(profile)
    if profile.role != "admin":
        raise IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)
    return profile


def get_current_profile(
    principal: IdentityPrincipal = Depends(verify_firebase_token),
    repository: FirestoreUserRepository = Depends(get_user_repository),
) -> UserProfile:
    """Load the durable profile for the Firebase identity on every request."""
    return current_profile(principal, repository)


def get_approved_user(
    profile: UserProfile = Depends(get_current_profile),
) -> UserProfile:
    """Authorize verified, approved users for protected application operations."""
    return require_approved_user(profile)


def get_admin_user(
    profile: UserProfile = Depends(get_current_profile),
) -> UserProfile:
    """Authorize the approved administrator role for administrative operations."""
    return require_admin(profile)
