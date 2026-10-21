from datetime import datetime, timezone

import pytest

from app.evaluation import InMemoryEvaluationRunStore
from app.release import (
    FirestoreReleaseDecisionStore,
    FirestoreReleaseManifestStore,
    ReleaseDecisionRequest,
    ReleaseDecisionService,
)
from test_evaluation_generations import FakeFirestore
from test_release_decision import EvidenceStore, completed_run, request


def test_manifest_and_decision_survive_repository_restart() -> None:
    client = FakeFirestore()
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(completed_run())
    decisions = FirestoreReleaseDecisionStore(client)
    service = ReleaseDecisionService(
        runs,
        decisions,
        EvidenceStore(),
        FirestoreReleaseManifestStore(client),
        actor_user_id="admin-1",
        clock=lambda: datetime(2026, 10, 21, tzinfo=timezone.utc),
    )

    stored = service.record(ReleaseDecisionRequest(**request()))
    restarted_decisions = FirestoreReleaseDecisionStore(client)
    restarted_manifests = FirestoreReleaseManifestStore(client)

    assert restarted_decisions.get(stored.release_id) == stored
    assert restarted_decisions.current_approved() == stored
    manifest = restarted_manifests.get(stored.release_id)
    assert manifest is not None
    assert manifest.candidate_digest
    assert client.data["release_decisions/release-1"]["actor_user_id"] == "admin-1"
    assert client.data["release_decisions/release-1"]["decided_at"] == "2026-10-21T00:00:00+00:00"


def test_decision_is_immutable_and_does_not_store_interaction_content() -> None:
    client = FakeFirestore()
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(completed_run())
    service = ReleaseDecisionService(
        runs,
        FirestoreReleaseDecisionStore(client),
        EvidenceStore(),
        FirestoreReleaseManifestStore(client),
        actor_user_id="admin-1",
    )
    service.record(ReleaseDecisionRequest(**request()))

    with pytest.raises(ValueError, match="already bound"):
        service.record(ReleaseDecisionRequest(**request(rationale="changed")))

    persisted = client.data["release_decisions/release-1"]
    assert "question" not in repr(persisted)
    assert "answer" not in repr(persisted)
    assert "secret" not in repr(persisted)
