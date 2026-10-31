from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.api.evaluation_reviews import (
    batch_review_repository,
    evaluation_generation_store,
    evaluation_run_store,
    pinned_chunk_source,
    support_evidence_store,
)
from app.contracts import EvaluationDataset, EvaluationItem, GoldAnnotation, UserProfile
from app.evaluation.batch_reviews import InMemoryBatchReviewRepository
from app.evaluation.support import InMemorySupportEvidenceStore, prepare_support
from app.identity import get_admin_user
from app.main import app
from test_evaluation_support import Source, chunk, run


def admin() -> UserProfile:
    return UserProfile(
        user_id="admin", email="admin@example.com", username="admin",
        approval_state="approved", role="admin", email_verified=True,
    )


@dataclass
class Runs:
    value: object

    def get(self, run_id: str):
        return self.value if run_id == "run-v1" else None


@dataclass
class Generations:
    dataset: EvaluationDataset

    def active(self):
        return type("Active", (), {"generation_id": self.dataset.dataset_version_id})()

    def active_dataset(self):
        return self.dataset


def fixture():
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[
            EvaluationItem(item_id="item-1", question="What is metadata?", dataset_version_id="dataset-v1"),
            EvaluationItem(item_id="item-2", question="What is a domain?", dataset_version_id="dataset-v1"),
        ],
        gold_annotations=[
            GoldAnnotation(item_id="item-1", expected_answer_or_rubric="Define it.", review_status="approved", supporting_passage="exact support"),
            GoldAnnotation(item_id="item-2", expected_answer_or_rubric="Explain it.", review_status="approved", supporting_passage="missing support"),
        ],
    )
    record = run(["item-1", "item-2"]).model_copy(update={"status": "completed"})
    source = Source((chunk("chunk-1", "exact support"),))
    evidence = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence)
    repository = InMemoryBatchReviewRepository()
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
        batch_review_repository: lambda: repository,
    })
    return record, repository


def test_review_projection_is_browser_safe_and_attestation_is_compare_and_set() -> None:
    _, repository = fixture()
    try:
        client = TestClient(app)
        response = client.get("/api/admin/evaluations/run-v1/review-summary")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["summary"]["status"] == "exception"
        assert payload["sample"][0]["question"] == "What is metadata?"
        assert payload["exceptions"][0]["reason"] == "missing_supporting_passage"
        forbidden = {"chunk_id", "candidate_chunk_ids", "content_hash", "candidate_snapshot_sha256", "corpus_version_id"}
        assert not forbidden.intersection(payload["sample"][0])
        assert not forbidden.intersection(payload["exceptions"][0])

        attested = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested", "exception_decisions": [{"item_id": "item-2", "decision": "inapplicable", "reason": "No support."}]},
        )
        assert attested.status_code == 200, attested.text
        assert attested.json()["status"] == "attested"
        repeated = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested", "exception_decisions": [{"item_id": "item-2", "decision": "inapplicable", "reason": "No support."}]},
        )
        assert repeated.status_code == 409
        assert len(repository._attestations) == 1
    finally:
        app.dependency_overrides.clear()


def test_review_write_rejects_technical_selectors_and_stale_run() -> None:
    record, _ = fixture()
    try:
        client = TestClient(app)
        malformed = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested", "corpus_version_id": "corpus-v2"},
        )
        assert malformed.status_code == 422

        record.status = "running"
        stale = client.get("/api/admin/evaluations/run-v1/review-summary")
        assert stale.status_code == 409
        assert "stale" not in stale.text.lower() or "not ready" in stale.text.lower()
    finally:
        app.dependency_overrides.clear()
