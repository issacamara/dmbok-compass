"""Exercise the shipped API and evaluation worker against one installed lock."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.api.evaluations import evaluation_service
from app.contracts import (
    CandidateIdentity,
    CorpusServingBinding,
    EvaluationDataset,
    EvaluationItem,
    UserProfile,
)
from app.evaluation import (
    EvaluationJobService,
    EvaluationOutcome,
    EvaluationWorker,
    InMemoryEvaluationDatasetStore,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
)
from app.identity import get_admin_user
from app.main import app
from app.rag import LangChainRagFactory, LangChainRagSettings, _dependency_lock_sha256


def _factory() -> LangChainRagFactory:
    # Snapshot construction needs settings, not a live Firestore or model client.
    factory = LangChainRagFactory.__new__(LangChainRagFactory)
    factory.settings = LangChainRagSettings(project="snapshot-smoke")
    return factory


class _SnapshotResolver:
    def resolve(
        self,
        *,
        corpus_version_id: str,
        configuration_version_id: str,
        model_version_id: str,
    ):
        binding = CorpusServingBinding(
            corpus_version_id=corpus_version_id,
            vector_collection="document_chunks_v2",
            vector_schema_version="2",
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
        )
        return _factory().candidate_snapshot(
            binding, request_timeout_ms=14_000, generation_timeout_ms=8_000
        )


def test_installed_lock_and_api_job_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.rag as rag

    lock = Path(rag.__file__).resolve().parents[1] / "requirements.lock"
    expected = os.environ.get("EXPECTED_LOCK_SHA256")
    if os.environ.get("FINAL_IMAGE_SMOKE") == "1":
        assert lock == Path("/app/requirements.lock")
        assert expected is not None
        manifest_lock = Path("/app/requirements.lock.sha256")
        assert manifest_lock.is_file()
        assert manifest_lock.read_text().split()[0] == expected
    assert lock.is_file()
    digest = sha256(lock.read_bytes()).hexdigest()
    assert digest == _dependency_lock_sha256()
    if expected is not None:
        assert digest == expected

    dataset = EvaluationDataset(
        dataset_version_id="snapshot-dataset",
        items=[EvaluationItem(item_id="item-1", question="Fixture question", dataset_version_id="snapshot-dataset")],
    )
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset)
    service = EvaluationJobService(
        datasets, InMemoryEvaluationRunStore(), InMemoryEvaluationJobDispatcher(), _SnapshotResolver()
    )
    admin = UserProfile(
        user_id="snapshot-admin", email="admin@example.com", username="snapshot-admin",
        approval_state="approved", role="admin", email_verified=True,
    )
    monkeypatch.setitem(app.dependency_overrides, get_admin_user, lambda: admin)
    monkeypatch.setitem(app.dependency_overrides, evaluation_service, lambda: service)
    response = TestClient(app).post(
        "/api/evaluations",
        json={
            "corpus_version_id": "snapshot-corpus",
            "configuration_version_id": "snapshot-config",
            "model_version_id": "snapshot-model",
        },
    )
    assert response.status_code == 202
    api_run = response.json()
    assert api_run["candidate_snapshot"]["dependency_lock_sha256"] == digest

    run = service.runs.get(api_run["run_id"])
    assert run is not None
    completed = EvaluationWorker(service.runs).execute(
        run.run_id,
        dataset,
        lambda _: EvaluationOutcome(True, True, True, True, True, 1.0),
    )
    assert completed.status == "completed"
    assert completed.candidate_snapshot is not None
    assert completed.candidate_snapshot.dependency_lock_sha256 == digest

    identity = CandidateIdentity(
        artifact_id=os.environ.get("IMAGE_ARTIFACT_ID", "local-snapshot-image"),
        installed_lock_sha256=digest,
        corpus_version_id=run.corpus_version_id,
        configuration_version_id=run.configuration_version_id,
        scorer_version=completed.candidate_snapshot.scorer_version,
    )
    assert identity.installed_lock_sha256 == digest
    print("snapshot-manifest=" + json.dumps(identity.model_dump(), sort_keys=True))


def test_missing_or_changed_installed_lock_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import app.rag as rag

    # Redirect only the module location so the production lock lookup is tested.
    module = tmp_path / "app" / "rag.py"
    module.parent.mkdir()
    monkeypatch.setattr(rag, "__file__", str(module))
    with pytest.raises(ValueError, match="candidate dependency lock is unavailable"):
        _dependency_lock_sha256()

    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"changed lock\n")
    assert _dependency_lock_sha256() == sha256(b"changed lock\n").hexdigest()
    assert _dependency_lock_sha256() != sha256(
        (Path(__file__).resolve().parents[1] / "requirements.lock").read_bytes()
    ).hexdigest()
