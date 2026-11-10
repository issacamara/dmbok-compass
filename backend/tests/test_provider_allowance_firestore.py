from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock

import pytest

from app.provider_allowance import ProviderAccountCapabilities, ProviderAllowanceObservation
from app.provider_allowance_firestore import FirestoreProviderAllowanceStore


NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


@dataclass
class FakeSnapshot:
    data: dict[str, object] | None

    @property
    def exists(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.data) if self.data is not None else {}


@dataclass
class FakeDocument:
    client: "FakeFirestore"
    path: str

    def get(self, *, transaction: "FakeTransaction | None" = None) -> FakeSnapshot:
        source = transaction.data if transaction is not None else self.client.data
        return FakeSnapshot(source.get(self.path))


@dataclass
class FakeCollection:
    client: "FakeFirestore"
    path: str

    def document(self, document_id: str) -> FakeDocument:
        return FakeDocument(self.client, f"{self.path}/{document_id}")


@dataclass
class FakeTransaction:
    client: "FakeFirestore"
    data: dict[str, dict[str, object]]

    def set(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.data[reference.path] = deepcopy(data)


@dataclass
class FakeFirestore:
    data: dict[str, dict[str, object]] = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)

    def collection(self, collection_id: str) -> FakeCollection:
        return FakeCollection(self, collection_id)

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self, deepcopy(self.data))


def run_fake_transaction(transaction: FakeTransaction, callback):  # type: ignore[no-untyped-def]
    with transaction.client.lock:
        # Refresh under the lock to model Firestore's transaction retry on
        # documents changed since the transaction's first read.
        transaction.data = deepcopy(transaction.client.data)
        result = callback(transaction)
        transaction.client.data = transaction.data
        return result


def account(kind="parsing", provider="llamacloud", account_id="dev"):
    return ProviderAccountCapabilities(provider, account_id, kind, max_units_per_operation=20, max_retries=2)


def observation(remaining=100, *, observed_at=NOW, expires_at=NOW + timedelta(hours=1)):
    return ProviderAllowanceObservation(remaining, observed_at, expires_at)


def store(client: FakeFirestore) -> FirestoreProviderAllowanceStore:
    return FirestoreProviderAllowanceStore(client, transaction_runner=run_fake_transaction)


def test_concurrent_transactional_reservations_cannot_overspend() -> None:
    client = FakeFirestore()
    caps = account()
    first_process = store(client)
    first_process.observe(caps, observation(10))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(
            lambda index: store(client).stage_reservation(caps, f"op-{index}", 2, retries=1, now=NOW),
            range(16),
        ))
    assert sum(result.status == "staged" for result in results) == 2
    account_record = next(value for key, value in client.data.items() if key.startswith("provider_allowance_accounts/"))
    assert account_record["reserved_units"] == 8


def test_older_or_conflicting_observation_cannot_replace_current_balance() -> None:
    client = FakeFirestore()
    caps = account()
    allowance_store = store(client)
    allowance_store.observe(caps, observation(5, observed_at=NOW))

    with pytest.raises(ValueError, match="newer allowance observation"):
        allowance_store.observe(caps, observation(100, observed_at=NOW - timedelta(minutes=1)))
    with pytest.raises(ValueError, match="conflicting allowance observations"):
        allowance_store.observe(caps, observation(100, observed_at=NOW))

    assert allowance_store.stage_reservation(caps, "stale-cannot-overspend", 6, now=NOW).status == "awaiting_budget"


def test_replaying_identical_observation_does_not_erase_reconciled_usage() -> None:
    client = FakeFirestore()
    caps = account()
    allowance_store = store(client)
    current_observation = observation(20)
    allowance_store.observe(caps, current_observation)
    assert allowance_store.stage_reservation(caps, "charged", 5, now=NOW).status == "staged"
    assert allowance_store.reconcile_usage(caps, "charged", 8).status == "staged"

    allowance_store.observe(caps, current_observation)
    assert allowance_store.stage_reservation(caps, "would-overspend", 13, now=NOW).status == "awaiting_budget"


def test_reservation_survives_store_restart_and_replay() -> None:
    client = FakeFirestore()
    caps = account()
    store(client).observe(caps, observation(20))
    original = store(client).stage_reservation(caps, "restart", 4, retries=1, now=NOW)

    restarted = store(client)
    replay = restarted.stage_reservation(caps, "restart", 19, now=NOW)
    assert replay.status == "staged"
    assert replay.reservation == original.reservation
    assert restarted.stage_reservation(caps, "restart", 4, now=NOW + timedelta(hours=2)).status == "awaiting_budget"


def test_unknown_actual_usage_keeps_transactional_hold() -> None:
    client = FakeFirestore()
    caps = account()
    allowance_store = store(client)
    allowance_store.observe(caps, observation(20))
    reservation = allowance_store.stage_reservation(caps, "unknown-actual", 5, now=NOW).reservation
    assert reservation is not None

    result = store(client).reconcile_usage(caps, "unknown-actual", None)
    assert result.status == "awaiting_budget"
    account_record = next(value for key, value in client.data.items() if key.startswith("provider_allowance_accounts/"))
    assert account_record["reserved_units"] == 5
    assert store(client).stage_reservation(caps, "unknown-actual", 5, now=NOW).status == "awaiting_budget"


def test_provider_and_kind_are_part_of_transaction_partition() -> None:
    client = FakeFirestore()
    parser = account("parsing", "llamacloud", "shared")
    judge = account("judging", "openai", "shared")
    allowance_store = store(client)
    allowance_store.observe(parser, observation(5))
    allowance_store.observe(judge, observation(5))
    assert allowance_store.stage_reservation(parser, "parser", 5, now=NOW).status == "staged"
    assert allowance_store.stage_reservation(judge, "judge", 5, now=NOW).status == "staged"
