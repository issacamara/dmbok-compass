"""Corpus-version lifecycle and active-pointer coordination.

The store in this module is deliberately backend-neutral.  Production callers
can map the same operations to Firestore transactions, while tests and the
offline ingestion job use :class:`InMemoryCorpusVersionStore`.
"""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Iterable, Protocol

from app.contracts import CorpusVersion, DocumentChunk


class CorpusVersionError(ValueError):
    """Base error for invalid corpus-version operations."""


class CorpusCompletenessError(CorpusVersionError):
    """Raised when a staged corpus is incomplete or internally inconsistent."""


class CorpusVersionStore(Protocol):
    def stage(self, version: CorpusVersion) -> CorpusVersion: ...

    def validate(self, version_id: str, chunks: Iterable[DocumentChunk]) -> CorpusVersion: ...

    def activate(self, version_id: str) -> CorpusVersion: ...

    def rollback(self, version_id: str | None = None) -> CorpusVersion: ...

    def get(self, version_id: str) -> CorpusVersion | None: ...

    def active(self) -> CorpusVersion | None: ...


class InMemoryCorpusVersionStore:
    """Reference implementation of the transactional version contract.

    A Firestore adapter can implement :class:`CorpusVersionStore` with one
    transaction around ``active_version`` and the affected version records.
    """

    def __init__(self) -> None:
        self._versions: dict[str, CorpusVersion] = {}
        self._active_id: str | None = None
        self._previous_active_id: str | None = None
        self._lock = RLock()

    def get(self, version_id: str) -> CorpusVersion | None:
        with self._lock:
            version = self._versions.get(version_id)
            return deepcopy(version) if version else None

    def active(self) -> CorpusVersion | None:
        with self._lock:
            return self.get(self._active_id) if self._active_id else None

    def stage(self, version: CorpusVersion) -> CorpusVersion:
        if version.status != "staged":
            raise CorpusVersionError("new corpus versions must start in staged state")
        with self._lock:
            existing = self._versions.get(version.version_id)
            if existing is not None:
                if existing.source_uri != version.source_uri:
                    raise CorpusVersionError("version_id is already bound to another source")
                return deepcopy(existing)
            self._versions[version.version_id] = version.model_copy(deep=True)
            return deepcopy(version)

    def validate(self, version_id: str, chunks: Iterable[DocumentChunk]) -> CorpusVersion:
        chunk_list = tuple(chunks)
        with self._lock:
            version = self._require(version_id)
            if version.status == "validated" or version.status == "active":
                return deepcopy(version)
            if version.status != "staged":
                raise CorpusVersionError(f"cannot validate a {version.status} corpus version")
            if not chunk_list:
                raise CorpusCompletenessError("corpus must contain at least one chunk")
            chunk_ids = [chunk.chunk_id for chunk in chunk_list]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise CorpusCompletenessError("corpus contains duplicate chunk IDs")
            if any(chunk.corpus_version_id != version_id for chunk in chunk_list):
                raise CorpusCompletenessError("every chunk must belong to the staged corpus version")
            if any(not chunk.text.strip() or len(chunk.embedding) != 768 for chunk in chunk_list):
                raise CorpusCompletenessError("every chunk must contain text and a 768-dimensional embedding")
            updated = version.model_copy(update={"status": "validated"}, deep=True)
            self._versions[version_id] = updated
            return deepcopy(updated)

    def activate(self, version_id: str) -> CorpusVersion:
        with self._lock:
            version = self._require(version_id)
            if version.status == "active" and self._active_id == version_id:
                return deepcopy(version)
            if version.status != "validated":
                raise CorpusVersionError("only a validated corpus version can become active")
            previous_id = self._active_id
            if previous_id is not None:
                previous = self._require(previous_id)
                self._versions[previous_id] = previous.model_copy(
                    update={"status": "retired", "created_at": previous.created_at}, deep=True
                )
            active = version.model_copy(update={"status": "active"}, deep=True)
            self._versions[version_id] = active
            self._active_id = version_id
            self._previous_active_id = previous_id
            return deepcopy(active)

    def rollback(self, version_id: str | None = None) -> CorpusVersion:
        with self._lock:
            target_id = version_id or self._previous_active_id
            if target_id is None:
                raise CorpusVersionError("no previous valid corpus version is available for rollback")
            target = self._require(target_id)
            if target.status not in {"retired", "active"}:
                raise CorpusVersionError("rollback target must be a previously valid corpus version")
            current_id = self._active_id
            if current_id == target_id:
                return deepcopy(target)
            if current_id is not None:
                current = self._require(current_id)
                self._versions[current_id] = current.model_copy(update={"status": "retired"}, deep=True)
            restored = target.model_copy(update={"status": "active"}, deep=True)
            self._versions[target_id] = restored
            self._active_id = target_id
            self._previous_active_id = current_id
            return deepcopy(restored)

    def _require(self, version_id: str) -> CorpusVersion:
        version = self._versions.get(version_id)
        if version is None:
            raise CorpusVersionError(f"unknown corpus version: {version_id}")
        return version
