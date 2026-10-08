"""Firestore-backed corpus-version metadata."""

from __future__ import annotations

from typing import Any

from app.contracts import CorpusVersion


class FirestoreCorpusVersionStore:
    """Persist corpus versions and resolve the single active version."""

    collection_id = "corpus_versions"

    def __init__(self, client: Any) -> None:
        self.client = client
        self.versions = client.collection(self.collection_id)

    def get(self, version_id: str) -> CorpusVersion | None:
        snapshot = self.versions.document(version_id).get()
        if not snapshot.exists:
            return None
        return CorpusVersion.model_validate(snapshot.to_dict())

    def active(self) -> CorpusVersion | None:
        snapshots = self.versions.where("status", "==", "active").limit(1).stream()
        snapshot = next(iter(snapshots), None)
        return CorpusVersion.model_validate(snapshot.to_dict()) if snapshot else None

    def stage(self, version: CorpusVersion) -> CorpusVersion:
        existing = self.get(version.version_id)
        if existing is not None:
            if existing.source_uri != version.source_uri:
                raise ValueError("version_id is already bound to another source")
            return existing
        self.versions.document(version.version_id).set(version.model_dump(mode="json"))
        return version

    def validate(self, version_id: str, chunks: Any) -> CorpusVersion:
        version = self.get(version_id)
        if version is None:
            raise ValueError(f"unknown corpus version: {version_id}")
        if version.status in {"validated", "active"}:
            return version
        if version.status != "staged":
            raise ValueError(f"cannot validate a {version.status} corpus version")
        validated = version.model_copy(update={"status": "validated"})
        self.versions.document(version_id).set(validated.model_dump(mode="json"))
        return validated

    def activate(self, version_id: str) -> CorpusVersion:
        version = self.get(version_id)
        if version is None:
            raise ValueError(f"unknown corpus version: {version_id}")
        if version.status == "active":
            return version
        if version.status != "validated":
            raise ValueError("only a validated corpus version can become active")
        batch = self.client.batch()
        for snapshot in self.versions.where("status", "==", "active").stream():
            batch.update(snapshot.reference, {"status": "retired"})
        active = version.model_copy(update={"status": "active"})
        batch.set(self.versions.document(version_id), active.model_dump(mode="json"))
        batch.commit()
        return active
