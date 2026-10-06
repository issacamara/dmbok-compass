from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.contracts.api import QuotaPolicy, UserProfile
from app.identity import IdentityPrincipal, get_user_repository, verify_firebase_token
from app.main import app, quota_repository
from app.quota import InMemoryQuotaRepository, QuotaExceededError


UTC_MIDNIGHT = datetime(2026, 10, 6, 23, 59, tzinfo=timezone.utc)


def test_quota_defaults_and_status_reset_at_next_utc_day() -> None:
    repository = InMemoryQuotaRepository()

    status = repository.status("user-1", UTC_MIDNIGHT)

    assert status.user_used == 0
    assert status.global_used == 0
    assert status.user_limit == 100
    assert status.global_limit == 100
    assert status.resets_at == datetime(2026, 10, 7, tzinfo=timezone.utc)


def test_reservation_increments_user_and_global_counters_atomically() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=2, global_daily_limit=3))

    first = repository.reserve("user-1", UTC_MIDNIGHT)
    second = repository.reserve("user-1", UTC_MIDNIGHT)
    other = repository.reserve("user-2", UTC_MIDNIGHT)

    assert (first.user_used, first.global_used) == (1, 1)
    assert (second.user_used, second.global_used) == (2, 2)
    assert (other.user_used, other.global_used) == (1, 3)


def test_user_limit_refusal_does_not_increment_global_counter() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=1, global_daily_limit=10))
    repository.reserve("user-1", UTC_MIDNIGHT)

    with pytest.raises(QuotaExceededError) as caught:
        repository.reserve("user-1", UTC_MIDNIGHT)

    assert caught.value.scope == "user"
    assert repository.status("user-1", UTC_MIDNIGHT).global_used == 1


def test_global_limit_applies_across_users() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=10, global_daily_limit=1))
    repository.reserve("user-1", UTC_MIDNIGHT)

    with pytest.raises(QuotaExceededError) as caught:
        repository.reserve("user-2", UTC_MIDNIGHT)

    assert caught.value.scope == "global"
    assert repository.status("user-2", UTC_MIDNIGHT).global_used == 1


def test_counters_start_again_on_the_next_utc_day() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=1, global_daily_limit=1))
    repository.reserve("user-1", UTC_MIDNIGHT)

    next_day = UTC_MIDNIGHT + timedelta(minutes=2)
    status = repository.reserve("user-1", next_day)

    assert (status.user_used, status.global_used) == (1, 1)


def test_concurrent_reservations_never_exceed_global_limit() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=10, global_daily_limit=3))

    def reserve(index: int):
        try:
            return repository.reserve(f"user-{index}", UTC_MIDNIGHT)
        except QuotaExceededError:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(reserve, range(10)))

    assert len([result for result in results if result is not None]) == 3
    assert repository.status("user-0", UTC_MIDNIGHT).global_used == 3


def test_quota_endpoint_returns_status_only_for_approved_user() -> None:
    repository = InMemoryQuotaRepository()

    profile = UserProfile(
        user_id="user-1",
        email="reader@example.com",
        username="Reader",
        approval_state="approved",
        role="user",
        email_verified=True,
    )
    app.dependency_overrides[verify_firebase_token] = lambda: IdentityPrincipal(
        user_id="user-1", email="reader@example.com", email_verified=True
    )
    app.dependency_overrides[get_user_repository] = lambda: _RepositoryWithProfile(profile)
    app.dependency_overrides[quota_repository] = lambda: repository
    try:
        response = TestClient(app).get("/api/quota")
        assert response.status_code == 200
        assert response.json()["user_limit"] == 100
        assert response.json()["global_used"] == 0
    finally:
        app.dependency_overrides.clear()


def test_quota_reservation_refuses_exhausted_quota_with_reset_details() -> None:
    repository = InMemoryQuotaRepository(QuotaPolicy(per_user_daily_limit=0, global_daily_limit=100))
    from app.main import reserve_request_quota

    profile = UserProfile(
        user_id="user-1",
        email="reader@example.com",
        username="Reader",
        approval_state="approved",
        role="user",
        email_verified=True,
    )
    with pytest.raises(QuotaExceededError) as caught:
        reserve_request_quota(profile=profile, repository=repository)

    assert caught.value.scope == "user"
    assert caught.value.status.resets_at.tzinfo is not None


class _RepositoryWithProfile:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile

    def refresh_verified_email(self, principal):
        return self.profile
