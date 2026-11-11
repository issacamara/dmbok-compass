"""Transactional reservations for external-provider usage allowances.

Observations and provider-enforced limits are deliberately separate: a limit
on one operation is not evidence of the remaining account balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock
from typing import Literal, Protocol

from app.contracts import AllowanceReservation

AllowanceKind = Literal["parsing", "generation", "judging"]
MAX_RETRY_COUNT = 3


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("allowance timestamps must include a timezone")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ProviderAllowanceObservation:
    """A frozen provider/account observation of remaining usage units."""

    remaining_units: int | None
    observed_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if self.remaining_units is not None and type(self.remaining_units) is not int:
            raise ValueError("remaining_units must be an integer or unknown")
        if self.remaining_units is not None and self.remaining_units < 0:
            raise ValueError("remaining_units cannot be negative")
        observed_at = _utc(self.observed_at)
        expires_at = _utc(self.expires_at)
        if expires_at <= observed_at:
            raise ValueError("allowance expiry must follow observation")
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class ProviderAccountCapabilities:
    """Non-secret provider/account identity and provider-enforced limits."""

    provider_id: str
    account_id: str
    kind: AllowanceKind
    max_units_per_operation: int | None = None
    max_retries: int = MAX_RETRY_COUNT

    def __post_init__(self) -> None:
        if not self.provider_id.strip() or not self.account_id.strip():
            raise ValueError("provider and account identifiers are required")
        if self.kind not in ("parsing", "generation", "judging"):
            raise ValueError("unsupported allowance kind")
        if self.max_units_per_operation is not None and type(self.max_units_per_operation) is not int:
            raise ValueError("provider operation limit must be an integer")
        if self.max_units_per_operation is not None and self.max_units_per_operation <= 0:
            raise ValueError("provider operation limit must be positive")
        if type(self.max_retries) is not int:
            raise ValueError("max_retries must be an integer")
        if not 0 <= self.max_retries <= MAX_RETRY_COUNT:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRY_COUNT}")


@dataclass(frozen=True)
class AllowanceStageResult:
    status: Literal["staged", "awaiting_budget"]
    reservation: AllowanceReservation | None
    reason: str | None = None


@dataclass(frozen=True)
class _ReservationState:
    record: AllowanceReservation
    status: Literal["reserved", "usage_unknown", "reconciled"] = "reserved"
    actual_units: int | None = None


class ProviderAllowanceStore(Protocol):
    def observe(
        self,
        capabilities: ProviderAccountCapabilities,
        observation: ProviderAllowanceObservation,
    ) -> None: ...

    def stage_reservation(
        self,
        capabilities: ProviderAccountCapabilities,
        operation_key: str,
        estimated_units: int,
        *,
        retries: int = 0,
        now: datetime | None = None,
    ) -> AllowanceStageResult: ...

    def reconcile_usage(
        self,
        capabilities: ProviderAccountCapabilities,
        operation_key: str,
        actual_units: int | None,
    ) -> AllowanceStageResult: ...


def _now(value: datetime | None) -> datetime:
    return _utc(value or datetime.now(timezone.utc))


def provider_allowance_reservation_id(
    capabilities: ProviderAccountCapabilities, operation_key: str
) -> str:
    """Return the stable identity binding a reservation to provider, account, kind and operation."""
    if not operation_key.strip():
        raise ValueError("operation_key is required")
    identity = (capabilities.provider_id, capabilities.account_id, capabilities.kind, operation_key)
    return sha256("\0".join(identity).encode("utf-8")).hexdigest()


def _bounded_retries(requested: int, capabilities: ProviderAccountCapabilities) -> int:
    if type(requested) is not int:
        raise ValueError("retries must be an integer")
    if requested < 0:
        raise ValueError("retries cannot be negative")
    return min(requested, capabilities.max_retries, MAX_RETRY_COUNT)


def _result_for_existing(state: _ReservationState, now: datetime) -> AllowanceStageResult:
    if state.status == "reserved" and now < state.record.allowance_expires_at:
        return AllowanceStageResult("staged", state.record)
    if now >= state.record.allowance_expires_at:
        reason = "reservation expired and requires sponsor reconciliation"
    elif state.status == "reconciled":
        reason = "operation is already reconciled; duplicate work is not authorized"
    else:
        reason = "usage is unresolved"
    return AllowanceStageResult("awaiting_budget", state.record, reason)


class InMemoryProviderAllowanceStore:
    """Thread-safe allowance ledger for deterministic tests and local use."""

    def __init__(self) -> None:
        self.observations: dict[tuple[str, str, AllowanceKind], tuple[ProviderAccountCapabilities, ProviderAllowanceObservation]] = {}
        self.reservations: dict[tuple[str, str, AllowanceKind, str], _ReservationState] = {}
        self.reserved_units: dict[tuple[str, str, AllowanceKind], int] = {}
        self.spent_units: dict[tuple[str, str, AllowanceKind], int] = {}
        self._lock = Lock()

    @staticmethod
    def _account_key(capabilities: ProviderAccountCapabilities) -> tuple[str, str, AllowanceKind]:
        return capabilities.provider_id, capabilities.account_id, capabilities.kind

    def observe(self, capabilities: ProviderAccountCapabilities, observation: ProviderAllowanceObservation) -> None:
        account_key = self._account_key(capabilities)
        with self._lock:
            previous = self.observations.get(account_key)
            if previous is not None:
                _, current = previous
                if observation.observed_at < current.observed_at:
                    raise ValueError("cannot replace a newer allowance observation")
                if observation.observed_at == current.observed_at:
                    if observation != current:
                        raise ValueError("conflicting allowance observations share a timestamp")
                    if previous[0] != capabilities:
                        raise ValueError("capabilities changed without a newer allowance observation")
                    return
            self.observations[account_key] = (capabilities, observation)
            # A fresh observed balance already reflects previously reconciled
            # usage. Unresolved reservations remain held against it.
            self.spent_units[account_key] = 0

    def stage_reservation(
        self,
        capabilities: ProviderAccountCapabilities,
        operation_key: str,
        estimated_units: int,
        *,
        retries: int = 0,
        now: datetime | None = None,
    ) -> AllowanceStageResult:
        if not operation_key.strip():
            raise ValueError("operation_key is required")
        if type(estimated_units) is not int:
            raise ValueError("estimated_units must be an integer")
        if estimated_units <= 0:
            raise ValueError("estimated_units must be positive")
        current = _now(now)
        account_key = self._account_key(capabilities)
        reservation_key = (*account_key, operation_key)
        with self._lock:
            existing = self.reservations.get(reservation_key)
            if existing is not None:
                return _result_for_existing(existing, current)
            observation_entry = self.observations.get(account_key)
            if observation_entry is None:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation is unknown")
            stored_capabilities, observation = observation_entry
            if observation.remaining_units is None:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation has unknown remaining units")
            if current < observation.observed_at or current >= observation.expires_at:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation is stale")
            if stored_capabilities != capabilities:
                return AllowanceStageResult("awaiting_budget", None, "provider capabilities changed; refresh observation")
            if capabilities.max_units_per_operation is not None and estimated_units > capabilities.max_units_per_operation:
                return AllowanceStageResult("awaiting_budget", None, "operation exceeds provider-enforced limit")
            retry_count = _bounded_retries(retries, capabilities)
            reserve_units = estimated_units * (1 + retry_count)
            held = self.reserved_units.get(account_key, 0)
            spent = self.spent_units.get(account_key, 0)
            if reserve_units > observation.remaining_units - held - spent:
                return AllowanceStageResult("awaiting_budget", None, "estimated usage exceeds unreserved allowance")
            record = AllowanceReservation(
                reservation_id=provider_allowance_reservation_id(capabilities, operation_key),
                operation_key=operation_key,
                kind=capabilities.kind,
                allowance_observed_at=observation.observed_at,
                allowance_expires_at=observation.expires_at,
                reserved_units=reserve_units,
            )
            self.reservations[reservation_key] = _ReservationState(record)
            self.reserved_units[account_key] = held + reserve_units
            return AllowanceStageResult("staged", record)

    def reconcile_usage(
        self,
        capabilities: ProviderAccountCapabilities,
        operation_key: str,
        actual_units: int | None,
    ) -> AllowanceStageResult:
        if actual_units is not None and actual_units < 0:
            raise ValueError("actual usage cannot be negative")
        if actual_units is not None and type(actual_units) is not int:
            raise ValueError("actual usage must be an integer or unknown")
        account_key = self._account_key(capabilities)
        reservation_key = (*account_key, operation_key)
        with self._lock:
            state = self.reservations.get(reservation_key)
            if state is None:
                return AllowanceStageResult("awaiting_budget", None, "reservation is unknown")
            if state.status == "reconciled":
                if actual_units is None or state.actual_units == actual_units:
                    return AllowanceStageResult("staged", state.record)
                return AllowanceStageResult("awaiting_budget", state.record, "actual usage conflicts with prior reconciliation")
            if actual_units is None:
                self.reservations[reservation_key] = _ReservationState(state.record, "usage_unknown")
                return AllowanceStageResult("awaiting_budget", state.record, "actual usage is unknown; reservation remains held")
            held = self.reserved_units.get(account_key, 0)
            self.reserved_units[account_key] = max(0, held - state.record.reserved_units)
            self.spent_units[account_key] = self.spent_units.get(account_key, 0) + actual_units
            self.reservations[reservation_key] = _ReservationState(state.record, "reconciled", actual_units)
            return AllowanceStageResult("staged", state.record)
