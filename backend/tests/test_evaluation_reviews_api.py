from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256

from fastapi.testclient import TestClient

from app.api.evaluation_reviews import (
    batch_review_repository,
    evaluation_generation_store,
    evaluation_run_store,
    evaluation_stage_store,
    pinned_chunk_source,
    support_evidence_store,
)
from app.api.evaluation_assessments import assessment_repository, output_store
from app.contracts import CandidateIdentity, EvaluationDataset, EvaluationItem, GoldAnnotation, ReviewedSemanticAssessment, UserProfile
from app.evaluation.batch_reviews import BatchReviewError, InMemoryBatchReviewRepository
from app.evaluation.job import EvaluationItemStage
from app.evaluation.reviews import InMemoryAssessmentRepository, RunItemAssessment
from app.evaluation.support import InMemorySupportEvidenceStore, StoredSupportChunk, prepare_support
from app.identity import get_admin_user, get_current_profile
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


@dataclass
class Generations:
    dataset: EvaluationDataset

    def active(self):
        return type("Active", (), {"generation_id": self.dataset.dataset_version_id})()

    def active_dataset(self):
        return self.dataset


class Stages:
    def __init__(self, stages):
        self.stages = {(stage.run_id, stage.item_id): stage for stage in stages}

    def get(self, run_id, item_id):
        return self.stages.get((run_id, item_id))


class Outputs:
    def __init__(self, audits):
        self.audits = {audit.attempt_id: audit for audit in audits}

    def get_audit(self, attempt_id):
        return self.audits.get(attempt_id)


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
    identity = CandidateIdentity(
        artifact_id="artifact", installed_lock_sha256="c" * 64,
        corpus_version_id=record.corpus_version_id,
        configuration_version_id=record.configuration_version_id,
        scorer_version=record.candidate_snapshot.scorer_version,
    )
    attempt_id = "attempt-item-1"
    output_digest = sha256(b"reviewed output").hexdigest()
    assessment = RunItemAssessment(
        generation_id=record.dataset_version_id, attempt_id=attempt_id, candidate_id="candidate-v1",
        scorer_version=identity.scorer_version, output_digest=output_digest,
        claim_support=True, citation_support=True, rubric_passed=True, refusal_judgment=False,
    )
    assessments = InMemoryAssessmentRepository()
    stored_assessment = assessments.save(assessment, actor=admin(), candidate_identity=identity, reviewed_at=datetime.now(timezone.utc))
    audit = type("Audit", (), {
        "evaluation_run_id": record.run_id, "evaluation_item_id": "item-1", "attempt_id": attempt_id,
        "candidate_id": assessment.candidate_id, "candidate_identity": identity,
        "scorer_version": assessment.scorer_version, "output_digest": output_digest,
    })()
    stages = Stages([EvaluationItemStage(
        run_id=record.run_id, item_id="item-1", output_handle=attempt_id, output_digest=output_digest,
    )])
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
        batch_review_repository: lambda: repository,
        assessment_repository: lambda: assessments,
        output_store: lambda: Outputs([audit]),
        evaluation_stage_store: lambda: stages,
    })
    return record, repository, stored_assessment, stages


def test_review_projection_is_browser_safe_and_attestation_is_compare_and_set() -> None:
    _, repository, assessment, _ = fixture()
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
            json={
                "decision": "attested",
                "exception_decisions": [{"item_id": "item-2", "decision": "inapplicable", "reason": "No support."}],
                "semantic_assessments": [{"item_id": "item-1", "assessment_id": assessment.assessment_id}],
            },
        )
        assert attested.status_code == 200, attested.text
        assert attested.json()["status"] == "attested"
        repeated = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={
                "decision": "attested",
                "exception_decisions": [{"item_id": "item-2", "decision": "inapplicable", "reason": "No support."}],
                "semantic_assessments": [{"item_id": "item-1", "assessment_id": assessment.assessment_id}],
            },
        )
        assert repeated.status_code == 409
        assert len(repository._attestations) == 1
    finally:
        app.dependency_overrides.clear()


def test_review_write_rejects_technical_selectors_and_stale_run() -> None:
    record, _, _, _ = fixture()
    try:
        client = TestClient(app)
        malformed = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested", "corpus_version_id": "corpus-v2"},
        )
        assert malformed.status_code == 422

        record.status = "running"
        stale = client.get("/api/admin/evaluations/run-v1/review-summary")
        individual_stale = client.get("/api/admin/evaluations/run-v1/items/item-1/review")
        assert stale.status_code == 409
        assert individual_stale.status_code == 409
        assert "stale" not in stale.text.lower() or "not ready" in stale.text.lower()
    finally:
        app.dependency_overrides.clear()


def test_review_routes_require_an_approved_administrator() -> None:
    try:
        client = TestClient(app)
        assert client.get("/api/admin/evaluations/run-v1/review-summary").status_code == 401
        assert client.get("/api/admin/evaluations/run-v1/items/item-1/review").status_code == 401
        assert client.put("/api/admin/evaluations/run-v1/items/item-1/review", json={"decision": "approve"}).status_code == 401

        app.dependency_overrides[get_current_profile] = lambda: UserProfile(
            user_id="member", email="member@example.com", username="member",
            approval_state="approved", role="user", email_verified=True,
        )
        assert client.get("/api/admin/evaluations/run-v1/review-summary").status_code == 403
        assert client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested"},
        ).status_code == 403
        assert client.get("/api/admin/evaluations/run-v1/items/item-1/review").status_code == 403
        assert client.put("/api/admin/evaluations/run-v1/items/item-1/review", json={"decision": "approve"}).status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_missing_and_stale_review_snapshots_fail_without_disclosing_bindings() -> None:
    record, repository, _, _ = fixture()
    try:
        client = TestClient(app)
        missing = client.get("/api/admin/evaluations/unknown-run/review-summary")
        assert missing.status_code == 404

        app.dependency_overrides[evaluation_generation_store] = lambda: type(
            "StaleGenerations", (), {
                "active": lambda self: type("Active", (), {"generation_id": "dataset-v2"})(),
                "active_dataset": lambda self: None,
            }
        )()
        stale = client.get("/api/admin/evaluations/run-v1/review-summary")
        assert stale.status_code == 409
        assert "dataset-v1" not in stale.text
        assert not repository._attestations
    finally:
        app.dependency_overrides.clear()


def test_partial_attestation_is_safe_conflict_and_does_not_write() -> None:
    _, repository, _, _ = fixture()
    try:
        client = TestClient(app)
        response = client.post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={"decision": "attested", "exception_decisions": []},
        )
        assert response.status_code == 409
        assert "incomplete_semantic_attestation" in response.text
        assert not repository._attestations
    finally:
        app.dependency_overrides.clear()


def test_batch_attestation_rejects_assessment_for_a_different_current_stage_output() -> None:
    _, repository, assessment, stages = fixture()
    stage = stages.get("run-v1", "item-1")
    stages.stages[("run-v1", "item-1")] = replace(stage, output_digest="e" * 64)
    try:
        response = TestClient(app).post(
            "/api/admin/evaluations/run-v1/batch-attestation",
            json={
                "decision": "attested",
                "exception_decisions": [{"item_id": "item-2", "decision": "inapplicable", "reason": "No support."}],
                "semantic_assessments": [{"item_id": "item-1", "assessment_id": assessment.assessment_id}],
            },
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "stale_semantic_assessment"
        assert not repository._attestations
    finally:
        app.dependency_overrides.clear()


def test_batch_review_revalidates_cross_child_spans_and_demotes_changed_child_evidence() -> None:
    parent_one, children_one = hierarchy(("one two three", "four five six"))
    parent_two, children_two = hierarchy(("alpha beta gamma", "delta epsilon zeta"))
    parent_two_id = f"{parent_two.parent_id}-second"
    parent_two = parent_two.model_copy(update={"parent_id": parent_two_id})
    children_two = tuple(child.model_copy(update={
        "parent_id": parent_two_id,
        "chunk_id": f"{child.chunk_id}-second",
    }) for child in children_two)
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[
            EvaluationItem(item_id="item-1", question="What spans the boundary?", dataset_version_id="dataset-v1"),
            EvaluationItem(item_id="item-2", question="What is alpha beta?", dataset_version_id="dataset-v1"),
        ],
        gold_annotations=[
            GoldAnnotation(item_id="item-1", expected_answer_or_rubric="Boundary", review_status="approved", supporting_passage="three four"),
            GoldAnnotation(item_id="item-2", expected_answer_or_rubric="Alpha", review_status="approved", supporting_passage="alpha beta"),
        ],
    )
    record = run(["item-1", "item-2"]).model_copy(update={"status": "completed"})
    records = tuple(
        StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent)
        for parent, children in ((parent_one, children_one), (parent_two, children_two))
        for child in children
    )
    source = Source(records)
    evidence = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence)
    batch_repository = InMemoryBatchReviewRepository()
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
        batch_review_repository: lambda: batch_repository,
    })
    try:
        client = TestClient(app)
        current = client.get("/api/admin/evaluations/run-v1/review-summary")
        assert current.status_code == 200, current.text
        assert current.json()["summary"]["eligible_count"] == 2

        changed_parent_one, changed_children_one = hierarchy(("one two replaced", "four five six"))
        changed_records = tuple(
            StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=changed_parent_one)
            for child in changed_children_one
        ) + tuple(
            record for record in records if record.parent and record.parent.parent_id != parent_one.parent_id
        )
        app.dependency_overrides[pinned_chunk_source] = lambda: Source(changed_records)
        app.dependency_overrides[batch_review_repository] = lambda: InMemoryBatchReviewRepository()
        stale = client.get("/api/admin/evaluations/run-v1/review-summary")
        assert stale.status_code == 200, stale.text
        assert stale.json()["summary"]["exception_count"] == 1
        assert stale.json()["exceptions"][0]["item_id"] == "item-1"
        assert evidence.get_evidence(run_id="run-v1", item_id="item-1").match_status == "eligible"
    finally:
        app.dependency_overrides.clear()


def test_review_errors_do_not_expose_evaluator_corpus_secret_or_provider_text(caplog) -> None:
    record, _, _, _ = fixture()
    private_values = (
        "evaluator-authored-secret",
        "corpus-private-passage",
        "service-secret-token",
        "raw-provider-response-body",
    )

    class FailingRepository:
        def create_or_get(self, review):
            raise BatchReviewError(" ".join(private_values))

    app.dependency_overrides[batch_review_repository] = FailingRepository
    try:
        response = TestClient(app).get("/api/admin/evaluations/run-v1/review-summary")
        assert response.status_code == 409
        assert "invalid_batch_review" in response.text
        for private_value in private_values:
            assert private_value not in response.text
            assert all(private_value not in record.getMessage() for record in caplog.records)
    finally:
        app.dependency_overrides.clear()


def test_individual_review_is_browser_safe_server_bound_and_single_decision() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    record = run(["item-1"]).model_copy(update={"status": "completed"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="What is this?", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(item_id="item-1", expected_answer_or_rubric="Define it.", review_status="candidate", supporting_passage="three four")],
    )
    evidence = InMemorySupportEvidenceStore()
    source = Source(tuple(
        StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent)
        for child in children
    ))
    prepare_support(record, dataset, source, evidence)
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/review"
        projection = client.get(url)
        assert projection.status_code == 200, projection.text
        payload = projection.json()
        assert payload["support_excerpt"] == "three four"
        assert payload["status"] == "eligible"
        assert not {"chunk_id", "candidate_chunk_ids", "content_hash", "corpus_version_id"}.intersection(payload)

        approved = client.put(url, json={"decision": "approve"})
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "review_approved"
        stored = evidence.get_evidence(run_id="run-v1", item_id="item-1")
        assert stored.match_status == "review_approved"
        assert [choice.chunk_id for choice in stored.selected_chunks] == stored.candidate_chunk_ids
        assert client.put(url, json={"decision": "approve"}).status_code == 409
        assert client.put(url, json={"decision": "approve", "selected_chunk_ids": ["other"]}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_ambiguous_support_cannot_be_approved_or_partially_selected() -> None:
    record = run(["item-1"]).model_copy(update={"status": "completed"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(item_id="item-1", expected_answer_or_rubric="A", review_status="candidate", supporting_passage="same passage")],
    )
    source = Source((chunk("first", "same passage"), chunk("second", "same passage", ordinal=1)))
    evidence = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence)
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/review"
        assert client.get(url).json()["status"] == "ambiguous"
        assert client.put(url, json={"decision": "approve"}).status_code == 409
        assert evidence.get_evidence(run_id=record.run_id, item_id="item-1").match_status == "ambiguous"
        rejected = client.put(url, json={"decision": "reject"})
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["status"] == "review_rejected"
        assert client.put(url, json={"decision": "reject"}).status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_explicit_unanswerable_approval_has_no_support_ids_and_stale_output_fails_closed() -> None:
    record = run(["item-1"]).model_copy(update={"status": "completed"})
    refusal = ReviewedSemanticAssessment(
        response_sha256=sha256(b"refusal").hexdigest(), scorer_version="evaluation-gates-v1", expected_refusal=True,
    )
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(item_id="item-1", expected_answer_or_rubric="Refuse", review_status="approved", semantic_assessment=refusal)],
    )
    source = Source(())
    evidence = InMemorySupportEvidenceStore()
    prepare_support(record, dataset, source, evidence)
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: source,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/review"
        assert client.get(url).json()["status"] == "inapplicable"
        response = client.put(url, json={"decision": "approve"})
        assert response.status_code == 200, response.text
        stored = evidence.get_evidence(run_id=record.run_id, item_id="item-1")
        assert stored.expected_refusal and stored.match_status == "review_approved"
        assert stored.selected_chunks == [] and stored.selected_chunk is None
        assert client.put(url, json={"decision": "approve"}).status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_changed_child_hash_invalidates_individual_review_without_a_write() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    record = run(["item-1"]).model_copy(update={"status": "completed"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(item_id="item-1", expected_answer_or_rubric="A", review_status="candidate", supporting_passage="three four")],
    )
    evidence = InMemorySupportEvidenceStore()
    original = Source(tuple(StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent) for child in children))
    prepare_support(record, dataset, original, evidence)
    changed = Source(tuple(
        StoredSupportChunk(
            child.model_copy(update={"text": child.text + " changed"}) if index == 0 else child,
            child.chunk_id,
            "d" * 64,
            parent=parent,
        )
        for index, child in enumerate(children)
    ))
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: lambda: Runs(record),
        evaluation_generation_store: lambda: Generations(dataset),
        support_evidence_store: lambda: evidence,
        pinned_chunk_source: lambda: changed,
    })
    try:
        client = TestClient(app)
        url = "/api/admin/evaluations/run-v1/items/item-1/review"
        assert client.put(url, json={"decision": "approve"}).status_code == 409
        assert evidence.get_evidence(run_id=record.run_id, item_id="item-1").match_status == "eligible"
    finally:
        app.dependency_overrides.clear()
