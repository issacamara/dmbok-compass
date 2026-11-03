from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.api.evaluation_support import evaluation_run_store, pinned_chunk_source, support_evidence_store
from app.contracts import EvaluationDataset, EvaluationItem, GoldAnnotation, UserProfile
from app.evaluation.support import InMemorySupportEvidenceStore, StoredSupportChunk, prepare_support
from app.identity import get_admin_user
from app.main import app
from test_evaluation_support import Source, chunk, hierarchy, run


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


def test_api_individually_approves_eligible_hierarchical_support() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    record = run(["item-1"])
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="three four",
        )],
    )
    source = Source(tuple(
        StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent)
        for child in children
    ))
    evidence_store = InMemorySupportEvidenceStore()
    prepared = prepare_support(record, dataset, source, evidence_store).evidence[0]
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        support_evidence_store: lambda: evidence_store,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/support"
        child_ids = prepared.candidate_chunk_ids
        approved = client.put(url, json={"decision": "approve", "selected_chunk_ids": child_ids})
        repeated = client.put(url, json={"decision": "approve", "selected_chunk_ids": child_ids})
    finally:
        app.dependency_overrides.clear()

    assert approved.status_code == 200, approved.text
    evidence = approved.json()["evidence"]
    assert evidence["match_status"] == "review_approved"
    assert [choice["chunk_id"] for choice in evidence["selected_chunks"]] == child_ids
    assert evidence["run_id"] == record.run_id
    assert evidence["corpus_version_id"] == record.corpus_version_id
    assert evidence["source_provenance_sha256"] == "d" * 64
    assert evidence["parent_id"] == parent.parent_id
    assert evidence["parent_content_hash"] == parent.content_hash
    assert evidence["parent_span_sha256"] == prepared.parent_span_sha256
    assert repeated.status_code == 200
    assert repeated.json()["evidence"] == evidence


def test_api_individually_rejects_eligible_hierarchical_support() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    record = run(["item-1"])
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="three four",
        )],
    )
    source = Source(tuple(
        StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent)
        for child in children
    ))
    evidence_store = InMemorySupportEvidenceStore()
    prepared = prepare_support(record, dataset, source, evidence_store).evidence[0]
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        support_evidence_store: lambda: evidence_store,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/support"
        rejected = client.put(url, json={"decision": "reject"})
        repeated = client.put(url, json={"decision": "reject"})
    finally:
        app.dependency_overrides.clear()

    assert prepared.match_status == "eligible"
    assert rejected.status_code == 200, rejected.text
    evidence = rejected.json()["evidence"]
    assert evidence["match_status"] == "review_rejected"
    assert evidence["selected_chunks"] == []
    assert evidence["parent_span_sha256"] == prepared.parent_span_sha256
    assert repeated.status_code == 200
    assert repeated.json()["evidence"] == evidence


def test_api_rejects_support_evidence_from_a_different_run_corpus() -> None:
    record = run(["item-1"])
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="duplicate",
        )],
    )
    source = Source((chunk("a", "duplicate"),))
    evidence_store = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence_store)
    mismatched_run = record.model_copy(update={"corpus_version_id": "corpus-v2"})
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(mismatched_run),
        support_evidence_store: lambda: evidence_store,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        response = client.put(
            "/api/admin/evaluations/run-v1/items/item-1/support",
            json={"decision": "reject"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "stale_evaluation_run"
