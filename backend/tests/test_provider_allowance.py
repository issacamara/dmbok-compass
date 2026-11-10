from datetime import datetime, timedelta, timezone

import pytest

from app.provider_allowance import (
    InMemoryProviderAllowanceStore,
    ProviderAccountCapabilities,
    ProviderAllowanceObservation,
)


NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


def account(kind="parsing", provider="llamacloud", account="dev"):
    return ProviderAccountCapabilities(provider, account, kind, max_units_per_operation=20, max_retries=2)


def observation(remaining=100, *, observed_at=NOW, expires_at=NOW + timedelta(hours=1)):
    return ProviderAllowanceObservation(remaining, observed_at, expires_at)


def test_unknown_stale_and_insufficient_observations_pause_for_budget() -> None:
    store = InMemoryProviderAllowanceStore()
    caps = account()
    assert store.stage_reservation(caps, "unknown", 1, now=NOW).status == "awaiting_budget"

    store.observe(caps, observation(None))
    assert store.stage_reservation(caps, "unknown-balance", 1, now=NOW).status == "awaiting_budget"

    store.observe(caps, observation(observed_at=NOW - timedelta(hours=2), expires_at=NOW - timedelta(hours=1)))
    assert store.stage_reservation(caps, "stale", 1, now=NOW).status == "awaiting_budget"

    store.observe(caps, observation(5))
    result = store.stage_reservation(caps, "too-large", 3, retries=1, now=NOW)
    assert result.status == "awaiting_budget"
    assert result.reservation is None


def test_concurrent_reservations_cannot_overspend_observed_balance() -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = InMemoryProviderAllowanceStore()
    caps = account()
    store.observe(caps, observation(10))
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda index: store.stage_reservation(caps, f"op-{index}", 2, retries=1, now=NOW), range(16)))
    assert sum(result.status == "staged" for result in results) == 2
    assert store.reserved_units[(caps.provider_id, caps.account_id, caps.kind)] == 8


def test_retry_estimate_is_bounded_and_provider_ceiling_is_per_operation() -> None:
    store = InMemoryProviderAllowanceStore()
    caps = account()
    store.observe(caps, observation(40))
    result = store.stage_reservation(caps, "bounded", 4, retries=999, now=NOW)
    assert result.status == "staged"
    assert result.reservation is not None
    assert result.reservation.reserved_units == 12

    assert store.stage_reservation(caps, "limit", 21, retries=0, now=NOW).status == "awaiting_budget"
    limited = ProviderAccountCapabilities("llamacloud", "other", "parsing", max_units_per_operation=2)
    store.observe(limited, observation(100))
    assert store.stage_reservation(limited, "operation-limit", 3, now=NOW).status == "awaiting_budget"


def test_unknown_usage_stays_held_and_known_usage_reconciles_once() -> None:
    store = InMemoryProviderAllowanceStore()
    caps = account()
    store.observe(caps, observation(20))
    reservation = store.stage_reservation(caps, "job", 5, now=NOW).reservation
    assert reservation is not None

    unresolved = store.reconcile_usage(caps, "job", None)
    assert unresolved.status == "awaiting_budget"
    assert store.reserved_units[(caps.provider_id, caps.account_id, caps.kind)] == 5
    assert store.stage_reservation(caps, "job", 5, now=NOW).status == "awaiting_budget"

    assert store.reconcile_usage(caps, "job", 3).status == "staged"
    assert store.reserved_units[(caps.provider_id, caps.account_id, caps.kind)] == 0
    assert store.spent_units[(caps.provider_id, caps.account_id, caps.kind)] == 3
    assert store.reconcile_usage(caps, "job", 3).status == "staged"
    assert store.reconcile_usage(caps, "job", 4).status == "awaiting_budget"


def test_replay_returns_original_reservation_and_expiry_never_authorizes_again() -> None:
    store = InMemoryProviderAllowanceStore()
    caps = account()
    store.observe(caps, observation(20, expires_at=NOW + timedelta(minutes=1)))
    original = store.stage_reservation(caps, "replay", 5, now=NOW)
    replay = store.stage_reservation(caps, "replay", 99, now=NOW)
    assert replay.status == "staged"
    assert replay.reservation == original.reservation

    expired = store.stage_reservation(caps, "replay", 5, now=NOW + timedelta(minutes=2))
    assert expired.status == "awaiting_budget"
    assert expired.reservation == original.reservation
    assert store.reserved_units[(caps.provider_id, caps.account_id, caps.kind)] == 5


def test_provider_and_operation_kinds_have_separate_budgets() -> None:
    store = InMemoryProviderAllowanceStore()
    parser = account(kind="parsing", provider="llamacloud", account="shared")
    judge = account(kind="judging", provider="openai", account="shared")
    store.observe(parser, observation(5))
    store.observe(judge, observation(5))
    assert store.stage_reservation(parser, "p", 5, now=NOW).status == "staged"
    assert store.stage_reservation(judge, "j", 5, now=NOW).status == "staged"


def test_timestamps_must_be_aware_and_ordered() -> None:
    with pytest.raises(ValueError, match="timezone"):
        ProviderAllowanceObservation(1, datetime(2026, 11, 10), NOW + timedelta(hours=1))
    with pytest.raises(ValueError, match="expiry"):
        observation(expires_at=NOW)
