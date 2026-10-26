from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest
from google.api_core.exceptions import AlreadyExists

from app.contracts import CandidateIdentity, UserProfile
from app.evaluation.candidate_outputs import (
    CandidateOutputAuthorizationError,
    CandidateOutputSourceError,
    FirestoreCandidateOutputStore,
    POLICY_ID,
)


@dataclass
class Snapshot:
    data: dict[str, object] | None

    @property
    def exists(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.data) if self.data is not None else {}


@dataclass
class Document:
    client: "Firestore"
    path: str

    def get(self) -> Snapshot:
        return Snapshot(self.client.data.get(self.path))

    def create(self, data: dict[str, object]) -> None:
        if self.path in self.client.data:
            raise AlreadyExists("already exists")
        self.client.data[self.path] = deepcopy(data)

    def set(self, data: dict[str, object]) -> None:
        self.client.data[self.path] = deepcopy(data)

    def delete(self) -> None:
        self.client.data.pop(self.path, None)


@dataclass
class Collection:
    client: "Firestore"
    name: str

    def document(self, document_id: str) -> Document:
        return Document(self.client, f"{self.name}/{document_id}")

    def stream(self) -> list[Snapshot]:
        prefix = f"{self.name}/"
        return [Snapshot(data) for path, data in self.client.data.items() if path.startswith(prefix)]


@dataclass
class Firestore:
    data: dict[str, dict[str, object]] = field(default_factory=dict)

    def collection(self, name: str) -> Collection:
        return Collection(self, name)


def admin() -> UserProfile:
    return UserProfile(
        user_id="admin", email="admin@example.com", username="admin",
        approval_state="approved", role="admin", email_verified=True,
    )


def user() -> UserProfile:
    return admin().model_copy(update={"user_id": "user", "role": "user"})


def identity() -> CandidateIdentity:
    return CandidateIdentity(
        artifact_id="artifact", installed_lock_sha256="a" * 64,
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )


def put(store: FirestoreCandidateOutputStore, *, generated_at: datetime) -> None:
    store.put(
        actor=admin(), evaluation_run_id="run-1", evaluation_item_id="item-1",
        candidate_id="candidate-1", attempt_id="attempt-1", candidate_identity=identity(),
        output="exact candidate output", generated_at=generated_at,
        source="administrator_authored_evaluation_question", scorer_version="scorer-v1",
    )


def test_raw_output_is_admin_only_and_finalization_retains_content_free_audit() -> None:
    client = Firestore()
    store = FirestoreCandidateOutputStore(client)
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    put(store, generated_at=now)

    with pytest.raises(CandidateOutputAuthorizationError):
        store.get_raw(actor=user(), attempt_id="attempt-1")
    assert store.get_raw(actor=admin(), attempt_id="attempt-1") == "exact candidate output"

    audit = store.finalize(actor=admin(), attempt_id="attempt-1", assessment={"passed": True})

    assert audit is not None
    assert audit.policy_id == POLICY_ID
    assert audit.output_digest
    assert store.get_raw(actor=admin(), attempt_id="attempt-1") is None
    saved_audit = client.data["evaluation_candidate_output_audit/attempt-1"]
    assert saved_audit["assessment"] == {"passed": True}
    assert "output" not in saved_audit


def test_production_source_is_rejected_without_creating_a_document() -> None:
    client = Firestore()
    store = FirestoreCandidateOutputStore(client)
    with pytest.raises(CandidateOutputSourceError):
        store.put(
            actor=admin(), evaluation_run_id="run-1", evaluation_item_id="item-1",
            candidate_id="candidate-1", attempt_id="attempt-1", candidate_identity=identity(),
            output="production answer", generated_at=datetime.now(timezone.utc),
            source="production_interaction",  # type: ignore[arg-type]
            scorer_version="scorer-v1",
        )
    assert client.data == {}


def test_expiry_cleanup_is_idempotent_and_marks_recovery() -> None:
    client = Firestore()
    store = FirestoreCandidateOutputStore(client)
    generated = datetime(2026, 9, 20, tzinfo=timezone.utc)
    put(store, generated_at=generated)

    result = store.cleanup_expired(actor=admin(), now=generated + timedelta(days=30), recovery=True)
    repeated = store.cleanup_expired(actor=admin(), now=generated + timedelta(days=31), recovery=True)

    assert result.deleted_attempt_ids == ("attempt-1",)
    assert result.recovery is True
    assert repeated.deleted_attempt_ids == ()
    assert store.get_raw(actor=admin(), attempt_id="attempt-1") is None
