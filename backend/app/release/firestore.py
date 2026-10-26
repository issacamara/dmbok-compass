"""Durable, immutable release manifests and sponsor decision records."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Protocol

from google.api_core.exceptions import AlreadyExists
from pydantic import Field

from app.contracts.api import ContractModel, ReleaseDecision


class ReleaseManifest(ContractModel):
    """The non-secret candidate identity that a release decision authorizes."""

    release_id: str = Field(min_length=1, max_length=128)
    evaluation_run_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    corpus_version_id: str = Field(min_length=1, max_length=128)
    configuration_version_id: str = Field(min_length=1, max_length=128)
    provider_version_id: str = Field(min_length=1, max_length=128)
    model_version_id: str = Field(min_length=1, max_length=128)
    scorer_version_id: str = Field(min_length=1, max_length=128)
    candidate_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    gate_report_ids: list[str] = Field(min_length=1, max_length=100)


class ReleaseManifestStore(Protocol):
    def create_or_get(self, manifest: ReleaseManifest) -> ReleaseManifest: ...

    def get(self, release_id: str) -> ReleaseManifest | None: ...


def candidate_digest(snapshot: object) -> str:
    """Hash only the validated, non-secret candidate snapshot."""

    if snapshot is None or not hasattr(snapshot, "model_dump_json"):
        return ""
    return sha256(snapshot.model_dump_json().encode("utf-8")).hexdigest()


class FirestoreReleaseManifestStore:
    collection_id = "release_manifests"

    def __init__(self, client: Any) -> None:
        self.manifests = client.collection(self.collection_id)

    def create_or_get(self, manifest: ReleaseManifest) -> ReleaseManifest:
        reference = self.manifests.document(manifest.release_id)
        try:
            reference.create(manifest.model_dump(mode="json"))
            return manifest.model_copy(deep=True)
        except AlreadyExists:
            snapshot = reference.get()
            if not snapshot.exists:
                raise
            stored = ReleaseManifest.model_validate(snapshot.to_dict())
            if stored != manifest:
                raise ValueError("release_id is already bound to another manifest")
            return stored

    def get(self, release_id: str) -> ReleaseManifest | None:
        snapshot = self.manifests.document(release_id).get()
        return ReleaseManifest.model_validate(snapshot.to_dict()) if snapshot.exists else None


class FirestoreReleaseDecisionStore:
    """Append-only decisions with a transactionally maintained approved pointer."""

    collection_id = "release_decisions"
    state_collection_id = "release_state"
    approved_pointer_id = "approved"

    def __init__(self, client: Any) -> None:
        self.decisions = client.collection(self.collection_id)
        self.pointer = client.collection(self.state_collection_id).document(self.approved_pointer_id)

    def create_or_get(
        self,
        decision: ReleaseDecision,
        *,
        actor_user_id: str,
        decided_at: datetime,
        candidate_digest: str,
    ) -> ReleaseDecision:
        if not actor_user_id:
            raise ValueError("authenticated actor is required")
        reference = self.decisions.document(decision.release_id)
        payload = {
            "decision": decision.model_dump(mode="json"),
            "actor_user_id": actor_user_id,
            "decided_at": decided_at.astimezone(timezone.utc).isoformat(),
            "candidate_digest": candidate_digest,
        }
        try:
            reference.create(payload)
        except AlreadyExists:
            snapshot = reference.get()
            if not snapshot.exists:
                raise
            stored = snapshot.to_dict()
            if stored.get("decision") != decision.model_dump(mode="json"):
                raise ValueError("release_id is already bound to another decision")
            return ReleaseDecision.model_validate(stored["decision"])
        if decision.decision == "approved":
            self.pointer.set({"release_id": decision.release_id, "candidate_digest": candidate_digest})
        return decision.model_copy(deep=True)

    def get(self, release_id: str) -> ReleaseDecision | None:
        snapshot = self.decisions.document(release_id).get()
        if not snapshot.exists:
            return None
        return ReleaseDecision.model_validate(snapshot.to_dict()["decision"])

    def current_approved(self) -> ReleaseDecision | None:
        pointer = self.pointer.get()
        if not pointer.exists:
            return None
        release_id = pointer.to_dict().get("release_id")
        return self.get(release_id) if isinstance(release_id, str) else None
