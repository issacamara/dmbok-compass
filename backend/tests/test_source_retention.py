from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.release.source_retention import (
    ConcurrentReferenceChange,
    RecoveryArtifact,
    RecoveryReferenceSnapshot,
    ReleaseRecoveryReference,
    ReferenceSnapshot,
    ReleaseSourceReference,
    RetentionPolicy,
    SourceGeneration,
    SourceRetentionError,
    inventory,
    inventory_recovery_artifacts,
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


def recovery_artifact(kind: str, identity: str, *, days_old: int = 60) -> RecoveryArtifact:
    return RecoveryArtifact(
        kind=kind,  # type: ignore[arg-type]
        identity=identity,
        content_sha256=(len(identity) % 15 + 1) * "a"
        + "0" * (63 - len(identity) % 15),
        created_at=NOW - timedelta(days=days_old),
    )


def test_recovery_inventory_protects_all_active_and_rollback_inputs() -> None:
    kinds = ("source", "parsing", "map", "approval", "dataset", "review")
    artifacts = tuple(recovery_artifact(kind, f"{kind}-identity") for kind in kinds)
    references = RecoveryReferenceSnapshot.from_references(
        tuple(
            ReleaseRecoveryReference(
                "active" if index % 2 == 0 else "rollback",
                f"corpus-{index}",
                artifact,
                "active" if index % 2 == 0 else "rollback",
            )
            for index, artifact in enumerate(artifacts)
        )
    )

    result = inventory_recovery_artifacts(artifacts, references, now=NOW)

    assert {item.kind for item in result.protected} == set(kinds)
    assert result.eligible == result.refused == ()
    assert result.dry_run is True


def test_recovery_inventory_requires_exact_rebuild_path_for_unreferenced_inputs() -> None:
    rebuildable = recovery_artifact("parsing", "gs://bucket/attempt/parse.json#7")
    failed_attempt = recovery_artifact("map", "gs://bucket/attempt/map.json#8")
    recent = recovery_artifact("review", "review/run/item", days_old=1)
    refs = RecoveryReferenceSnapshot.from_references(())

    result = inventory_recovery_artifacts(
        (rebuildable, failed_attempt, recent),
        refs,
        now=NOW,
        rebuildable=(rebuildable.key,),
    )

    assert result.eligible == (rebuildable,)
    assert result.refused == (failed_attempt,)
    assert result.protected == (recent,)


def test_recovery_inventory_fails_closed_on_reference_change_or_unvalidated_record() -> None:
    item = recovery_artifact("dataset", "dataset/corpus/revision-4")
    first = RecoveryReferenceSnapshot.from_references((
        ReleaseRecoveryReference("release", "corpus", item, "active"),
    ))
    changed = RecoveryReferenceSnapshot.from_references(())

    class ChangingReader:
        def snapshot(self) -> RecoveryReferenceSnapshot:
            return changed

    with pytest.raises(ConcurrentReferenceChange):
        inventory_recovery_artifacts((item,), first, now=NOW, reader=ChangingReader())

    invalid = RecoveryReferenceSnapshot.from_references((
        ReleaseRecoveryReference("release", "corpus", item, "active", validated=False),
    ))
    with pytest.raises(SourceRetentionError, match="unvalidated recovery references"):
        inventory_recovery_artifacts((item,), invalid, now=NOW)
    with pytest.raises(SourceRetentionError, match="only supports dry-run"):
        inventory_recovery_artifacts((), changed, now=NOW, dry_run=False)
