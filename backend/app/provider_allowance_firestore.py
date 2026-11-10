"""Firestore implementation of the provider allowance transaction boundary."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Callable, TypeVar

from google.cloud import firestore

from app.contracts import AllowanceReservation
from app.provider_allowance import (
    AllowanceStageResult,
    ProviderAccountCapabilities,
    ProviderAllowanceObservation,
    _ReservationState,
    _bounded_retries,
    _now,
    _result_for_existing,
)

T = TypeVar("T")
_TransactionRunner = Callable[[Any, Callable[[Any], T]], T]


def _run_transaction(transaction: Any, callback: Callable[[Any], T]) -> T:
    @firestore.transactional
    def run(active_transaction: Any) -> T:
        return callback(active_transaction)

    return run(transaction)


def _digest(*parts: str) -> str:
    return sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _observation_payload(observation: ProviderAllowanceObservation) -> dict[str, Any]:
    return {
        "remaining_units": observation.remaining_units,
        "observed_at": observation.observed_at.isoformat(),
        "expires_at": observation.expires_at.isoformat(),
    }


def _read_observation(data: dict[str, Any]) -> ProviderAllowanceObservation:
    return ProviderAllowanceObservation(
        remaining_units=data.get("remaining_units"),
        observed_at=datetime.fromisoformat(data["observed_at"]),
        expires_at=datetime.fromisoformat(data["expires_at"]),
    )


def _read_reservation(data: dict[str, Any]) -> _ReservationState:
    return _ReservationState(
        record=AllowanceReservation.model_validate(data["reservation"]),
        status=data["status"],
        actual_units=data.get("actual_units"),
    )


class FirestoreProviderAllowanceStore:
    """Persist observations and reserve/reconcile units in Firestore transactions."""

    account_collection_id = "provider_allowance_accounts"
    reservation_collection_id = "provider_allowance_reservations"

    def __init__(self, client: Any, *, transaction_runner: _TransactionRunner = _run_transaction) -> None:
        self.client = client
        self.accounts = client.collection(self.account_collection_id)
        self.reservations = client.collection(self.reservation_collection_id)
        self._transaction_runner = transaction_runner

    @staticmethod
    def _account_id(capabilities: ProviderAccountCapabilities) -> str:
        return _digest(capabilities.provider_id, capabilities.account_id, capabilities.kind)

    @staticmethod
    def _reservation_id(capabilities: ProviderAccountCapabilities, operation_key: str) -> str:
        return _digest(capabilities.provider_id, capabilities.account_id, capabilities.kind, operation_key)

    def observe(self, capabilities: ProviderAccountCapabilities, observation: ProviderAllowanceObservation) -> None:
        account_ref = self.accounts.document(self._account_id(capabilities))

        def apply(transaction: Any) -> None:
            snapshot = account_ref.get(transaction=transaction)
            previous = snapshot.to_dict() if snapshot.exists else {}
            transaction.set(account_ref, {
                "provider_id": capabilities.provider_id,
                "account_id": capabilities.account_id,
                "kind": capabilities.kind,
                "capabilities": {
                    "max_units_per_operation": capabilities.max_units_per_operation,
                    "max_retries": capabilities.max_retries,
                },
                "observation": _observation_payload(observation),
                "reserved_units": int(previous.get("reserved_units", 0)),
                # A fresh provider observation includes reconciled prior usage.
                "spent_units": 0,
            })

        self._transaction_runner(self.client.transaction(), apply)

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
        account_ref = self.accounts.document(self._account_id(capabilities))
        reservation_ref = self.reservations.document(self._reservation_id(capabilities, operation_key))

        def apply(transaction: Any) -> AllowanceStageResult:
            account_snapshot = account_ref.get(transaction=transaction)
            reservation_snapshot = reservation_ref.get(transaction=transaction)
            if reservation_snapshot.exists:
                return _result_for_existing(_read_reservation(reservation_snapshot.to_dict()), current)
            if not account_snapshot.exists:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation is unknown")
            account = account_snapshot.to_dict()
            observation_data = account.get("observation")
            if not isinstance(observation_data, dict):
                return AllowanceStageResult("awaiting_budget", None, "allowance observation is unknown")
            observation = _read_observation(observation_data)
            if observation.remaining_units is None:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation has unknown remaining units")
            if current < observation.observed_at or current >= observation.expires_at:
                return AllowanceStageResult("awaiting_budget", None, "allowance observation is stale")
            stored_capabilities = account.get("capabilities", {})
            if stored_capabilities != {
                "max_units_per_operation": capabilities.max_units_per_operation,
                "max_retries": capabilities.max_retries,
            }:
                return AllowanceStageResult("awaiting_budget", None, "provider capabilities changed; refresh observation")
            if capabilities.max_units_per_operation is not None and estimated_units > capabilities.max_units_per_operation:
                return AllowanceStageResult("awaiting_budget", None, "operation exceeds provider-enforced limit")
            reserve_units = estimated_units * (1 + _bounded_retries(retries, capabilities))
            held = int(account.get("reserved_units", 0))
            spent = int(account.get("spent_units", 0))
            if reserve_units > observation.remaining_units - held - spent:
                return AllowanceStageResult("awaiting_budget", None, "estimated usage exceeds unreserved allowance")
            reservation = AllowanceReservation(
                reservation_id=self._reservation_id(capabilities, operation_key),
                operation_key=operation_key,
                kind=capabilities.kind,
                allowance_observed_at=observation.observed_at,
                allowance_expires_at=observation.expires_at,
                reserved_units=reserve_units,
            )
            transaction.set(account_ref, {**account, "reserved_units": held + reserve_units})
            transaction.set(reservation_ref, {"reservation": reservation.model_dump(mode="json"), "status": "reserved"})
            return AllowanceStageResult("staged", reservation)

        return self._transaction_runner(self.client.transaction(), apply)

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
        account_ref = self.accounts.document(self._account_id(capabilities))
        reservation_ref = self.reservations.document(self._reservation_id(capabilities, operation_key))

        def apply(transaction: Any) -> AllowanceStageResult:
            account_snapshot = account_ref.get(transaction=transaction)
            reservation_snapshot = reservation_ref.get(transaction=transaction)
            if not reservation_snapshot.exists:
                return AllowanceStageResult("awaiting_budget", None, "reservation is unknown")
            state = _read_reservation(reservation_snapshot.to_dict())
            if state.status == "reconciled":
                if actual_units is None or state.actual_units == actual_units:
                    return AllowanceStageResult("staged", state.record)
                return AllowanceStageResult("awaiting_budget", state.record, "actual usage conflicts with prior reconciliation")
            if actual_units is None:
                transaction.set(reservation_ref, {"reservation": state.record.model_dump(mode="json"), "status": "usage_unknown"})
                return AllowanceStageResult("awaiting_budget", state.record, "actual usage is unknown; reservation remains held")
            if not account_snapshot.exists:
                return AllowanceStageResult("awaiting_budget", state.record, "allowance account state is unavailable")
            account = account_snapshot.to_dict()
            account["reserved_units"] = max(0, int(account.get("reserved_units", 0)) - state.record.reserved_units)
            account["spent_units"] = int(account.get("spent_units", 0)) + actual_units
            transaction.set(account_ref, account)
            transaction.set(reservation_ref, {
                "reservation": state.record.model_dump(mode="json"),
                "status": "reconciled",
                "actual_units": actual_units,
            })
            return AllowanceStageResult("staged", state.record)

        return self._transaction_runner(self.client.transaction(), apply)
