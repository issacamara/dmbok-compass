from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.api.evaluation_support import evaluation_run_store, pinned_chunk_source, support_evidence_store
from app.contracts import EvaluationDataset, EvaluationItem, GoldAnnotation, UserProfile
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


def test_support_status_lists_server_candidates_and_review_is_idempotent() -> None:
    record = run(["item-1"])
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            supporting_passage="duplicate",
        )],
    )
    source = Source((chunk("a", "duplicate"), chunk("b", "duplicate", ordinal=1)))
    evidence_store = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence_store)
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        support_evidence_store: lambda: evidence_store,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/support"
        listed = client.get(url)
        approved = client.put(url, json={"decision": "approve", "selected_chunk_id": "a"})
        repeated = client.put(url, json={"decision": "approve", "selected_chunk_id": "a"})
        unlisted = client.put(url, json={"decision": "approve", "selected_chunk_id": "other"})
    finally:
        app.dependency_overrides.clear()

    assert listed.status_code == 200, listed.text
    assert [item["chunk_id"] for item in listed.json()["candidate_chunks"]] == ["a", "b"]
    assert approved.status_code == 200, approved.text
    assert approved.json()["evidence"]["match_status"] == "review_approved"
    assert repeated.status_code == 200
    assert repeated.json()["evidence"] == approved.json()["evidence"]
    assert unlisted.status_code == 409
