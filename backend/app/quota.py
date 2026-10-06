"""Transactional per-user and global UTC-day request quotas."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from threading import Lock
from typing import Protocol

from firebase_admin import firestore
from google.cloud import firestore as google_firestore

from app.contracts.api import QuotaPolicy, QuotaStatus
from app.identity import _firebase_app

DEFAULT_QUOTA_POLICY = QuotaPolicy()


class QuotaExceededError(Exception):
    """Raised when either the user or global daily quota is exhausted."""

    def __init__(self, status: QuotaStatus, scope: str) -> None:
        self.status = status
        self.scope = scope
        limit = status.user_limit if scope == "user" else status.global_limit
        super().__init__(f"The {scope} daily quota of {limit} requests has been reached.")


class QuotaRepository(Protocol):
    def get_policy(self) -> QuotaPolicy: ...

    def update_policy(self, policy: QuotaPolicy) -> QuotaPolicy: ...

    def status(self, user_id: str, now: datetime | None = None) -> QuotaStatus: ...

    def reserve(self, user_id: str, now: datetime | None = None) -> QuotaStatus: ...


def _utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def quota_day(now: datetime | None = None) -> date:
    return _utc_now(now).date()


def next_reset(now: datetime | None = None) -> datetime:
    current = _utc_now(now)
    return datetime.combine(current.date() + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _status(policy: QuotaPolicy, user_used: int, global_used: int, now: datetime) -> QuotaStatus:
    return QuotaStatus(
        user_used=user_used,
        user_limit=policy.per_user_daily_limit,
        global_used=global_used,
        global_limit=policy.global_daily_limit,
        resets_at=next_reset(now),
    )


class InMemoryQuotaRepository:
    """Deterministic repository for tests and local development."""

    def __init__(self, policy: QuotaPolicy = DEFAULT_QUOTA_POLICY) -> None:
        self.policy = policy
        self.user_counts: dict[tuple[date, str], int] = {}
        self.global_counts: dict[date, int] = {}
        self._lock = Lock()

    def get_policy(self) -> QuotaPolicy:
        return self.policy

    def update_policy(self, policy: QuotaPolicy) -> QuotaPolicy:
        with self._lock:
            self.policy = policy
            return policy

    def status(self, user_id: str, now: datetime | None = None) -> QuotaStatus:
        current = _utc_now(now)
        day = current.date()
        with self._lock:
            return _status(
                self.policy,
                self.user_counts.get((day, user_id), 0),
                self.global_counts.get(day, 0),
                current,
            )

    def reserve(self, user_id: str, now: datetime | None = None) -> QuotaStatus:
        current = _utc_now(now)
        day = current.date()
        with self._lock:
            current_status = _status(
                self.policy,
                self.user_counts.get((day, user_id), 0),
                self.global_counts.get(day, 0),
                current,
            )
            if current_status.user_used >= current_status.user_limit:
                raise QuotaExceededError(current_status, "user")
            if current_status.global_used >= current_status.global_limit:
                raise QuotaExceededError(current_status, "global")
            self.user_counts[(day, user_id)] = current_status.user_used + 1
            self.global_counts[day] = current_status.global_used + 1
            return _status(self.policy, current_status.user_used + 1, current_status.global_used + 1, current)


class FirestoreQuotaRepository:
    """Firestore-backed quota policy and counters using one transaction."""

    def __init__(self) -> None:
        self.client = firestore.client(app=_firebase_app())
        self.configuration = self.client.collection("application_configuration").document("quota_policy")
        self.user_counters = self.client.collection("user_quota_counters")
        self.global_counters = self.client.collection("global_quota_counters")

    def get_policy(self) -> QuotaPolicy:
        snapshot = self.configuration.get()
        if not snapshot.exists:
            return DEFAULT_QUOTA_POLICY
        return QuotaPolicy.model_validate(snapshot.to_dict())

    def update_policy(self, policy: QuotaPolicy) -> QuotaPolicy:
        self.configuration.set(policy.model_dump())
        return policy

    def _read_count(self, collection, key: str, transaction=None) -> int:  # type: ignore[no-untyped-def]
        snapshot = collection.document(key).get(transaction=transaction)
        return int(snapshot.get("count")) if snapshot.exists else 0

    def status(self, user_id: str, now: datetime | None = None) -> QuotaStatus:
        current = _utc_now(now)
        day = current.date().isoformat()
        return _status(
            self.get_policy(),
            self._read_count(self.user_counters, f"{day}:{user_id}"),
            self._read_count(self.global_counters, day),
            current,
        )

    def reserve(self, user_id: str, now: datetime | None = None) -> QuotaStatus:
        current = _utc_now(now)
        day = current.date().isoformat()
        user_ref = self.user_counters.document(f"{day}:{user_id}")
        global_ref = self.global_counters.document(day)
        transaction = self.client.transaction()

        @google_firestore.transactional
        def reserve_in_transaction(transaction):  # type: ignore[no-untyped-def]
            policy_snapshot = self.configuration.get(transaction=transaction)
            policy = (
                QuotaPolicy.model_validate(policy_snapshot.to_dict())
                if policy_snapshot.exists
                else DEFAULT_QUOTA_POLICY
            )
            user_used = self._read_count(self.user_counters, f"{day}:{user_id}", transaction)
            global_used = self._read_count(self.global_counters, day, transaction)
            current_status = _status(policy, user_used, global_used, current)
            if user_used >= policy.per_user_daily_limit:
                raise QuotaExceededError(current_status, "user")
            if global_used >= policy.global_daily_limit:
                raise QuotaExceededError(current_status, "global")
            transaction.set(user_ref, {"day": day, "user_id": user_id, "count": user_used + 1})
            transaction.set(global_ref, {"day": day, "count": global_used + 1})
            return _status(policy, user_used + 1, global_used + 1, current)

        return reserve_in_transaction(transaction)
