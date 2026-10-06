from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.contracts.api import UserProfile
from app.identity import (
    APPROVED_USER_LIMIT,
    IdentityError,
    IdentityPrincipal,
    _next_approved_count,
    get_user_repository,
    require_approved_user,
    verify_firebase_token,
)
from app.main import app


class MemoryUserRepository:
    def __init__(self) -> None:
        self.profiles: dict[str, UserProfile] = {}
        self.approved_count = 0

    def get(self, user_id: str) -> UserProfile | None:
        return self.profiles.get(user_id)

    def refresh_verified_email(self, principal: IdentityPrincipal) -> UserProfile | None:
        profile = self.profiles.get(principal.user_id)
        if profile is None:
            return None
        updated = profile.model_copy(
            update={"email": principal.email, "email_verified": profile.email_verified or principal.email_verified}
        )
        self.profiles[principal.user_id] = updated
        return updated

    def register(self, principal: IdentityPrincipal, username: str) -> UserProfile:
        existing = self.profiles.get(principal.user_id)
        if existing:
            return self.refresh_verified_email(principal)  # type: ignore[return-value]
        profile = UserProfile(
            user_id=principal.user_id,
            email=principal.email,
            username=username,
            approval_state="pending",
            role="user",
            email_verified=principal.email_verified,
        )
        self.profiles[principal.user_id] = profile
        return profile

    def list_profiles(self, approval_state: str | None, limit: int) -> list[UserProfile]:
        users = list(self.profiles.values())
        if approval_state:
            users = [user for user in users if user.approval_state == approval_state]
        return users[:limit]

    def set_approval(self, user_id: str, approval_state: str) -> UserProfile:
        profile = self.profiles.get(user_id)
        if profile is None:
            raise IdentityError("User profile was not found.", 404)
        if approval_state == "approved" and not profile.email_verified:
            raise IdentityError("The user must verify their email before approval.")
        self.approved_count = _next_approved_count(
            self.approved_count,
            profile.approval_state == "approved",
            approval_state == "approved",
        )
        updated = profile.model_copy(update={"approval_state": approval_state})
        self.profiles[user_id] = updated
        return updated


@pytest.fixture
def identity_client() -> Iterator[tuple[TestClient, MemoryUserRepository]]:
    repository = MemoryUserRepository()
    principal = IdentityPrincipal(user_id="firebase-user-1", email="reader@example.com", email_verified=False)
    app.dependency_overrides[verify_firebase_token] = lambda: principal
    app.dependency_overrides[get_user_repository] = lambda: repository  # type: ignore[assignment]
    try:
        yield TestClient(app), repository
    finally:
        app.dependency_overrides.clear()


def test_registration_uses_firebase_claims_and_creates_pending_profile(identity_client) -> None:
    client, repository = identity_client
    response = client.post("/api/registration", json={"username": "Reader"})

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "firebase-user-1",
        "email": "reader@example.com",
        "username": "Reader",
        "approval_state": "pending",
        "role": "user",
        "email_verified": False,
    }
    assert repository.profiles["firebase-user-1"].role == "user"


def test_registration_rejects_client_supplied_identity_fields(identity_client) -> None:
    client, _ = identity_client

    response = client.post(
        "/api/registration",
        json={"username": "Reader", "email": "attacker@example.com", "role": "admin"},
    )

    assert response.status_code == 422


def test_identity_routes_require_a_bearer_token() -> None:
    response = TestClient(app).get("/api/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_registration_status_and_profile_refresh_verification(identity_client) -> None:
    client, _ = identity_client
    client.post("/api/registration", json={"username": "Reader"})
    app.dependency_overrides[verify_firebase_token] = lambda: IdentityPrincipal(
        user_id="firebase-user-1", email="reader@example.com", email_verified=True
    )

    status_response = client.get("/api/registration/status")
    profile_response = client.get("/api/me")

    assert status_response.json() == {
        "registered": True,
        "approval_state": "pending",
        "email_verified": True,
    }
    assert profile_response.json()["email_verified"] is True


def test_only_approved_admin_can_list_and_change_profiles(identity_client) -> None:
    client, repository = identity_client
    client.post("/api/registration", json={"username": "Reader"})
    denied = client.get("/api/admin/users")
    assert denied.status_code == 403

    repository.profiles["admin"] = UserProfile(
        user_id="admin",
        email="admin@example.com",
        username="Admin",
        approval_state="approved",
        role="admin",
        email_verified=True,
    )
    app.dependency_overrides[verify_firebase_token] = lambda: IdentityPrincipal(
        user_id="admin", email="admin@example.com", email_verified=True
    )
    listed = client.get("/api/admin/users?approval_state=pending")
    assert listed.status_code == 200
    assert [user["user_id"] for user in listed.json()] == ["firebase-user-1"]

    repository.profiles["firebase-user-1"] = repository.profiles["firebase-user-1"].model_copy(
        update={"email_verified": True}
    )
    approved = client.patch("/api/admin/users/firebase-user-1", json={"approval_state": "approved"})
    assert approved.status_code == 200
    assert approved.json()["approval_state"] == "approved"
    assert repository.approved_count == 1


def test_admin_cannot_approve_unverified_user_or_exceed_the_limit(identity_client) -> None:
    client, repository = identity_client
    client.post("/api/registration", json={"username": "Reader"})
    repository.profiles["admin"] = UserProfile(
        user_id="admin",
        email="admin@example.com",
        username="Admin",
        approval_state="approved",
        role="admin",
        email_verified=True,
    )
    app.dependency_overrides[verify_firebase_token] = lambda: IdentityPrincipal(
        user_id="admin", email="admin@example.com", email_verified=True
    )

    unverified = client.patch("/api/admin/users/firebase-user-1", json={"approval_state": "approved"})
    assert unverified.status_code == 409

    repository.profiles["firebase-user-1"] = repository.profiles["firebase-user-1"].model_copy(
        update={"email_verified": True}
    )
    repository.approved_count = APPROVED_USER_LIMIT
    full = client.patch("/api/admin/users/firebase-user-1", json={"approval_state": "approved"})
    assert full.status_code == 409
    assert "limit" in full.json()["detail"]


def test_approved_access_gate_rejects_pending_and_unverified_profiles() -> None:
    pending = UserProfile(
        user_id="reader",
        email="reader@example.com",
        username="Reader",
        approval_state="pending",
        email_verified=True,
    )
    with pytest.raises(IdentityError) as error:
        require_approved_user(pending)
    assert error.value.status_code == 403

    unverified = pending.model_copy(update={"approval_state": "approved", "email_verified": False})
    with pytest.raises(IdentityError) as error:
        require_approved_user(unverified)
    assert error.value.status_code == 403


def test_approved_user_counter_changes_only_on_state_transitions() -> None:
    assert _next_approved_count(4, False, True) == 5
    assert _next_approved_count(4, True, True) == 4
    assert _next_approved_count(4, True, False) == 3
