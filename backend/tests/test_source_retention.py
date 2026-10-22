from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.release.source_retention import (
    ConcurrentReferenceChange,
    ReferenceSnapshot,
    ReleaseSourceReference,
    RetentionPolicy,
    SourceGeneration,
    SourceRetentionError,
    inventory,
)


NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


def source(generation: int, *, days_old: int = 60, digest: str | None = None) -> SourceGeneration:
    return SourceGeneration(
        bucket="corpus",
        object_name="source.pdf",
        generation=generation,
        content_sha256=digest or f"{generation:064x}",
        created_at=NOW - timedelta(days=days_old),
    )


def reference(item: SourceGeneration, release_id: str = "release-active") -> ReleaseSourceReference:
    return ReleaseSourceReference(release_id, f"corpus-{item.generation}", item, "active")


def test_active_and_rollback_generations_are_protected_by_exact_identity() -> None:
    active = source(1)
    rollback = source(2)
    unrelated = source(3)
    refs = ReferenceSnapshot.from_references((reference(active), reference(rollback, "release-rollback")))

    result = inventory((active, rollback, unrelated), refs, now=NOW, rebuildable=(unrelated.key,))

    assert {item.generation for item in result.protected} == {1, 2}
    assert [item.generation for item in result.eligible] == [3]
    assert result.dry_run is True


def test_unreferenced_source_without_verified_recovery_path_is_refused() -> None:
    item = source(1)

    result = inventory((item,), ReferenceSnapshot.from_references(()), now=NOW)

    assert result.refused == (item,)
    assert result.eligible == ()


def test_explicit_generation_and_additional_retention_are_protected() -> None:
    explicit = source(1)
    recent = source(2, days_old=1)
    result = inventory(
        (explicit, recent),
        ReferenceSnapshot.from_references(()),
        now=NOW,
        explicit_generations=(("corpus", "source.pdf", 1),),
        policy=RetentionPolicy(additional_retained_generations=1),
    )

    assert {item.generation for item in result.protected} == {1, 2}


@dataclass
class ChangingReader:
    snapshot_after: ReferenceSnapshot

    def snapshot(self) -> ReferenceSnapshot:
        return self.snapshot_after


def test_reference_change_aborts_inventory() -> None:
    first = ReferenceSnapshot.from_references((reference(source(1)),))
    changed = ReferenceSnapshot.from_references((reference(source(2)),))

    with pytest.raises(ConcurrentReferenceChange):
        inventory((source(1), source(2)), first, now=NOW, reader=ChangingReader(changed))


def test_inventory_rejects_non_dry_run_and_unvalidated_reference() -> None:
    item = source(1)
    invalid = ReleaseSourceReference("release", "corpus", item, "active", validated=False)
    refs = ReferenceSnapshot.from_references((invalid,))

    with pytest.raises(SourceRetentionError, match="unvalidated"):
        inventory((item,), refs, now=NOW)
    with pytest.raises(SourceRetentionError, match="only supports dry-run"):
        inventory((), ReferenceSnapshot.from_references(()), now=NOW, dry_run=False)
