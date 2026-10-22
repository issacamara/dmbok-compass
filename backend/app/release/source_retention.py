"""Release-aware, dry-run planning for versioned corpus source retention.

This module only plans work. It has no storage delete or infrastructure-apply
operation by design. A caller must re-read the release references before any
future, separately approved deletion operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
from typing import Iterable, Protocol


_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class SourceRetentionError(ValueError):
    """Raised when a safe retention inventory cannot be produced."""


class ConcurrentReferenceChange(SourceRetentionError):
    """Raised when release references change during an inventory."""


@dataclass(frozen=True, slots=True)
class SourceGeneration:
    bucket: str
    object_name: str
    generation: int
    content_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.bucket or self.bucket.strip() != self.bucket:
            raise SourceRetentionError("source bucket must be a normalized non-empty name")
        if not self.object_name or self.object_name.strip() != self.object_name:
            raise SourceRetentionError("source object name must be a normalized non-empty path")
        if self.generation < 1:
            raise SourceRetentionError("source generation must be positive")
        if not _SHA256.fullmatch(self.content_sha256):
            raise SourceRetentionError("source content hash must be a lowercase SHA-256 digest")
        if self.created_at.tzinfo is None:
            raise SourceRetentionError("source creation time must be timezone-aware")

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.bucket, self.object_name, self.generation, self.content_sha256)


@dataclass(frozen=True, slots=True)
class ReleaseSourceReference:
    release_id: str
    corpus_version_id: str
    source: SourceGeneration
    role: str
    validated: bool = True


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    references: tuple[ReleaseSourceReference, ...]
    fingerprint: str

    @classmethod
    def from_references(cls, references: Iterable[ReleaseSourceReference]) -> "ReferenceSnapshot":
        ordered = tuple(sorted(references, key=lambda item: (item.release_id, item.source.key)))
        material = json.dumps(
            [
                {
                    "release_id": item.release_id,
                    "corpus_version_id": item.corpus_version_id,
                    "source": item.source.key,
                    "role": item.role,
                    "validated": item.validated,
                }
                for item in ordered
            ],
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(ordered, sha256(material.encode("utf-8")).hexdigest())


class ReleaseReferenceReader(Protocol):
    def snapshot(self) -> ReferenceSnapshot: ...


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    max_age_days: int = 30
    additional_retained_generations: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.max_age_days <= 3650:
            raise SourceRetentionError("max_age_days must be between 1 and 3650")
        if not 0 <= self.additional_retained_generations <= 100:
            raise SourceRetentionError("additional_retained_generations must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class RetentionInventory:
    dry_run: bool
    reference_fingerprint: str
    protected: tuple[SourceGeneration, ...]
    eligible: tuple[SourceGeneration, ...]
    refused: tuple[SourceGeneration, ...]


def inventory(
    sources: Iterable[SourceGeneration],
    references: ReferenceSnapshot,
    *,
    now: datetime,
    policy: RetentionPolicy = RetentionPolicy(),
    explicit_generations: Iterable[tuple[str, str, int]] = (),
    rebuildable: Iterable[tuple[str, str, int, str]] = (),
    reader: ReleaseReferenceReader | None = None,
    dry_run: bool = True,
) -> RetentionInventory:
    """Return a bounded inventory without mutating any backing service.

    ``rebuildable`` is an explicit, verified recovery-path allowlist. An old
    object without that exact source identity is refused rather than treated as
    safe to collect. The optional reader provides a second reference snapshot
    to detect concurrent release changes.
    """

    if not dry_run:
        raise SourceRetentionError("source retention only supports dry-run inventories")
    if now.tzinfo is None:
        raise SourceRetentionError("now must be timezone-aware")
    if any(not reference.validated for reference in references.references):
        raise SourceRetentionError("unvalidated release references cannot protect source generations")

    source_list = tuple(sources)
    protected_keys = {reference.source.key for reference in references.references}
    protected_keys.update(
        (bucket, object_name, generation, "")
        for bucket, object_name, generation in explicit_generations
    )
    rebuildable_keys = {tuple(item) for item in rebuildable}
    cutoff = now.astimezone(timezone.utc) - timedelta(days=policy.max_age_days)
    old = tuple(sorted(
        (source for source in source_list if source.created_at.astimezone(timezone.utc) < cutoff),
        key=lambda source: source.created_at,
    ))

    protected: list[SourceGeneration] = []
    candidates: list[SourceGeneration] = []
    refused: list[SourceGeneration] = []
    for source in old:
        exact = source.key in protected_keys
        explicit = (source.bucket, source.object_name, source.generation, "") in protected_keys
        if exact or explicit:
            protected.append(source)
        elif source.key in rebuildable_keys:
            candidates.append(source)
        else:
            refused.append(source)

    if policy.additional_retained_generations:
        recent = sorted((source for source in source_list if source not in old), key=lambda item: item.created_at, reverse=True)
        protected.extend(recent[: policy.additional_retained_generations])

    if reader is not None:
        after = reader.snapshot()
        if after.fingerprint != references.fingerprint:
            raise ConcurrentReferenceChange("release references changed during source inventory")

    return RetentionInventory(
        dry_run=True,
        reference_fingerprint=references.fingerprint,
        protected=tuple(sorted(set(protected), key=lambda source: source.key)),
        eligible=tuple(sorted(candidates, key=lambda source: source.key)),
        refused=tuple(sorted(refused, key=lambda source: source.key)),
    )
