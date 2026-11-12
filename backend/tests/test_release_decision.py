from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.contracts import (
    CorpusServingBinding,
    EvaluationCandidateSnapshot,
    EvaluationMetric,
    EvaluationReportEligibility,
    EvaluationRun,
    DurableBatchAttestation,
    DurableExceptionDecision,
    DurableExceptionRecord,
    DurableRepresentativeSampleEntry,
    DurableReviewSummary,
    DurableSemanticAssessmentReference,
    RunBoundSupportEvidence,
    VerifiedChunkChoice,
    UserProfile,
)
from app.contracts.v29 import CandidateDatasetBinding, DatasetHead, EvaluationRowV2, EvaluationRunSnapshot, RawSourceSpan
from app.evaluation import InMemoryEvaluationRunStore
from app.identity import IdentityError, get_admin_user
from app.main import app, release_decision_service
from app.evaluation.batch_reviews import BatchReview
from app.evaluation.job import EvaluationItemStage
from app.evaluation.support import candidate_snapshot_sha256
from app.release import InMemoryReleaseDecisionStore, ReleaseDecisionRequest, ReleaseDecisionService
from app.release.firestore import candidate_digest


REQUIRED_METRICS = (
    "retrieval_success", "grounded_claims", "citation_correctness",
    "answer_quality", "refusal_correctness", "response_time",
)


class EvidenceStore:
    def __init__(self, eligibility: EvaluationReportEligibility | None = None) -> None:
        self.eligibility = eligibility or EvaluationReportEligibility(
            generation_id="dataset-v1", eligibility="release_evidence", approved_gold_count=40,
            is_current_generation=True, is_superseded=False,
        )
        self.run_snapshots: dict[str, tuple[EvaluationRunSnapshot, EvaluationCandidateSnapshot]] = {}
        self.heads: dict[str, DatasetHead] = {}

    def report_eligibility(self, generation_id: str) -> EvaluationReportEligibility:
        return self.eligibility

    def attach_run(self, run: EvaluationRun, *, answerable_count: int = 20) -> None:
        rows = []
        for index, item_id in enumerate(run.selected_item_ids):
            if index < answerable_count:
                rows.append(EvaluationRowV2(
                    item_id=item_id, origin="admin", question=f"Question {item_id}", category="definitions",
                    expected_outcome="answer", expected_refusal=False, expected_answer_or_rubric="Answer",
                    source_spans=[RawSourceSpan(
                        span_id=f"span-{item_id}", block_id=f"block-{item_id}", physical_page=1,
                        start_offset=0, end_offset=1, raw_text="x",
                    )], relevant_child_ids=[f"chunk-{item_id}"], parent_ids=[f"parent-{item_id}"],
                ))
            else:
                rows.append(EvaluationRowV2(
                    item_id=item_id, origin="admin", question=f"Question {item_id}", category="definitions",
                    expected_outcome="evidence_refusal", expected_refusal=True,
                    expected_answer_or_rubric="Refuse outside scope", scope_rationale="Outside corpus scope",
                ))
        snapshot = EvaluationRunSnapshot(
            run_id=run.run_id,
            binding=CandidateDatasetBinding(attempt_id="attempt-v1", corpus_version_id=run.corpus_version_id, dataset_revision=7),
            configuration_version_id=run.configuration_version_id,
            selection="full", items=tuple(rows), snapshot_sha256="f" * 64,
        )
        self.run_snapshots[run.run_id] = (snapshot, run.candidate_snapshot)
        self.heads[run.corpus_version_id] = DatasetHead(
            corpus_version_id=run.corpus_version_id, revision=7, dataset_sha256="e" * 64
        )

    def get_run_snapshot(self, run_id: str) -> tuple[EvaluationRunSnapshot, EvaluationCandidateSnapshot]:
        assert run_id in self.run_snapshots
        return self.run_snapshots[run_id]

    def corpus_head(self, corpus_version_id: str) -> DatasetHead | None:
        return self.heads.get(corpus_version_id)


class AttestedReviewStore:
    def __init__(
        self, run: EvaluationRun, *, decision: str = "attested", valid: bool = True,
        semantic_assessment_binding: str = "exact_output",
    ) -> None:
        snapshot = run.candidate_snapshot
        assert snapshot is not None
        digest = candidate_digest(snapshot)
        sample_count = min(len(run.selected_item_ids), min(10, max(5, (len(run.selected_item_ids) + 4) // 5)))
        sample_ids = run.selected_item_ids[:sample_count]
        self.review = BatchReview(
            summary=DurableReviewSummary(
                generation_id=run.dataset_version_id,
                run_id=run.run_id,
                candidate_snapshot_sha256=digest,
                corpus_version_id=run.corpus_version_id,
                status="eligible",
                eligible_count=len(run.selected_item_ids),
                exception_count=0,
                sampled_item_ids=sample_ids,
                decision_version="batch-review-v1",
            ),
            samples=tuple(DurableRepresentativeSampleEntry(
                generation_id=run.dataset_version_id,
                run_id=run.run_id,
                candidate_snapshot_sha256=digest,
                corpus_version_id=run.corpus_version_id,
                sample_identity=f"sample-{index}",
                item_id=item_id,
                candidate_chunk_ids=[f"chunk-{item_id}"],
                content_hashes=["a" * 64],
                provenance_digests=["b" * 64],
                status="eligible",
                decision_version="batch-review-v1",
            ) for index, item_id in enumerate(sample_ids)),
            exceptions=(),
        )
        self.result = DurableBatchAttestation(
            generation_id=run.dataset_version_id,
            run_id=run.run_id,
            candidate_snapshot_sha256=digest if valid else "f" * 64,
            corpus_version_id=run.corpus_version_id,
            sampled_item_ids=sample_ids,
            decision=decision,
            actor="admin",
            decided_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
            decision_version="batch-review-v1",
            semantic_assessment_binding=semantic_assessment_binding,
            semantic_assessments=[DurableSemanticAssessmentReference(
                item_id=item_id,
                assessment_id=f"assessment-{item_id}",
                output_digest="a" * 64,
                scorer_version=snapshot.scorer_version,
                reviewer_actor="reviewer",
                reviewed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
            ) for item_id in sample_ids] if semantic_assessment_binding == "exact_output" else [],
        )

    def get(self, *, generation_id: str, run_id: str) -> BatchReview | None:
        return self.review if (generation_id, run_id) == (self.review.summary.generation_id, self.review.summary.run_id) else None

    def attestation(self, review: BatchReview) -> DurableBatchAttestation | None:
        return self.result if review == self.review else None


class SupportEvidenceStore:
    def __init__(self, run: EvaluationRun, *, approved_count: int | None = None) -> None:
        self.run = run
        self.approved_count = len(run.selected_item_ids) if approved_count is None else approved_count
        self.marker = SimpleNamespace(
            run_id=run.run_id,
            corpus_version_id=run.corpus_version_id,
            candidate_snapshot_sha256=candidate_snapshot_sha256(run),
            item_count=len(run.selected_item_ids),
            prepared_item_count=len(run.selected_item_ids),
        )

    def get_marker(self, *, run_id: str) -> object | None:
        return self.marker if run_id == self.run.run_id else None

    def get_evidence(self, *, run_id: str, item_id: str) -> RunBoundSupportEvidence | None:
        if run_id != self.run.run_id or item_id not in self.run.selected_item_ids:
            return None
        if self.run.selected_item_ids.index(item_id) >= self.approved_count:
            return None
        if self.run.selected_item_ids.index(item_id) >= 20:
            return RunBoundSupportEvidence(
                run_id=self.run.run_id, item_id=item_id,
                candidate_snapshot_sha256=candidate_snapshot_sha256(self.run),
                corpus_version_id=self.run.corpus_version_id, source_provenance_sha256="b" * 64,
                match_status="review_approved", expected_refusal=True, reviewer_actor="admin",
                reviewed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
            )
        choice = VerifiedChunkChoice(
            chunk_id=f"chunk-{item_id}", corpus_version_id=self.run.corpus_version_id,
            content_hash="a" * 64, page=1, section="Section", source_provenance_sha256="b" * 64,
        )
        return RunBoundSupportEvidence(
            run_id=self.run.run_id, item_id=item_id,
            candidate_snapshot_sha256=candidate_snapshot_sha256(self.run),
            corpus_version_id=self.run.corpus_version_id, source_provenance_sha256="b" * 64,
            match_status="review_approved", candidate_chunk_ids=[choice.chunk_id],
            verified_chunks=[choice], selected_chunk=choice, selected_chunks=[choice],
            reviewer_actor="admin", reviewed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        )


class StageStore:
    def __init__(self, run: EvaluationRun, *, stale_output: bool = False) -> None:
        self.stages = {
            item_id: EvaluationItemStage(
                run_id=run.run_id, item_id=item_id, retrieval_status="succeeded",
                generation_status="succeeded", assessment_status="reviewed",
                output_handle="output" if stale_output else None,
                output_digest="b" * 64 if stale_output else "a" * 64,
                reviewed_output_digest="a" * 64,
            )
            for item_id in run.selected_item_ids
        }

    def get(self, run_id: str, item_id: str) -> EvaluationItemStage | None:
        del run_id
        return self.stages.get(item_id)


def completed_run(*, passed: bool = True, item_count: int = 40) -> EvaluationRun:
    return EvaluationRun(
        run_id="eval-1",
        dataset_version_id="dataset-v1",
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        candidate_snapshot=EvaluationCandidateSnapshot(
            corpus_binding=CorpusServingBinding(
                corpus_version_id="corpus-v1", vector_collection="chunks-v2",
                vector_schema_version="2", embedding_model="gemini-embedding-001",
                ingestion_fingerprint="a" * 64, configuration_fingerprint="b" * 64,
            ),
            project="project", location="europe-west1", answer_provider="openrouter",
            primary_model="model-v1", json_schema_supported=True,
            request_timeout_ms=1000, generation_timeout_ms=1000,
            prompt_version="prompt-v1", retrieval_policy_version="retrieval-v1",
            evidence_policy_version="evidence-v1", scorer_version="scorer-v1",
            dependency_lock_sha256="c" * 64, integration_artifact="wheel-v1",
            runtime_version="image@sha256:" + "d" * 64,
        ),
        status="completed",
        selected_item_ids=[f"item-{index}" for index in range(item_count)],
        report_eligibility=EvaluationReportEligibility(
            generation_id="dataset-v1",
            eligibility="release_evidence" if 40 <= item_count <= 50 else "exploratory",
            approved_gold_count=item_count,
            is_current_generation=True,
            is_superseded=False,
        ),
        metrics=[EvaluationMetric(
            metric_name=name, numerator=item_count if passed else 0, denominator=item_count,
            percentage=100 if passed else 0,
            threshold={"retrieval_success": 90, "grounded_claims": 95,
                       "citation_correctness": 95, "answer_quality": 85,
                       "refusal_correctness": 95, "response_time": 95}[name],
            passed=passed,
        ) for name in REQUIRED_METRICS],
    )


def request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "release_id": "release-1", "evaluation_run_id": "eval-1",
        "dataset_version_id": "dataset-v1", "corpus_version_id": "corpus-v1",
        "configuration_version_id": "config-v1", "provider_version_id": "openrouter-v1",
        "model_version_id": "model-v1", "scorer_version_id": "scorer-v1",
        "gate_report_ids": ["gate-report-1"], "decision": "approved", "rationale": "All gates passed.",
    }
    payload.update(overrides)
    return payload


_DEFAULT_REVIEWS = object()


def service(
    run: EvaluationRun,
    evidence: EvidenceStore | None = None,
    *,
    batch_reviews: object = _DEFAULT_REVIEWS,
) -> ReleaseDecisionService:
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(run)
    evidence_store = evidence or EvidenceStore()
    evidence_store.attach_run(run)
    return ReleaseDecisionService(
        runs, InMemoryReleaseDecisionStore(), evidence_store,
        batch_reviews=(AttestedReviewStore(run) if batch_reviews is _DEFAULT_REVIEWS else batch_reviews),  # type: ignore[arg-type]
        support_evidence=SupportEvidenceStore(run),
        stages=StageStore(run),
    )


def test_decision_binds_exact_evidence_and_retry_is_idempotent() -> None:
    release_service = service(completed_run())
    first = release_service.record(ReleaseDecisionRequest(**request()))
    result = release_service.record(ReleaseDecisionRequest(**request()))
    assert result == first
    assert result.provider_version_id == "openrouter-v1"
    assert release_service.get("release-1") == first


def test_stale_or_missing_evidence_is_rejected() -> None:
    release_service = service(completed_run())
    with pytest.raises(ValueError, match="stale"):
        release_service.record(ReleaseDecisionRequest(**request(dataset_version_id="dataset-old")))
    with pytest.raises(ValidationError):
        ReleaseDecisionRequest(**request(gate_report_ids=[]))


def test_approval_requires_the_complete_candidate_binding() -> None:
    run = completed_run()
    release_service = service(run)
    with pytest.raises(ValueError, match="scorer"):
        release_service.record(ReleaseDecisionRequest(**request(scorer_version_id="legacy-scorer")))
    with pytest.raises(ValueError, match="provider"):
        release_service.record(ReleaseDecisionRequest(**request(provider_version_id="vertex-v1")))


def test_approval_rejects_proxy_or_inconsistent_metric_scoring() -> None:
    run = completed_run().model_copy(update={
        "metrics": [run_metric.model_copy(update={"threshold": 90}) for run_metric in completed_run().metrics]
    })
    with pytest.raises(ValueError, match="incompatible metric scoring"):
        service(run).record(ReleaseDecisionRequest(**request()))


def test_release_id_cannot_be_reused_for_different_evidence() -> None:
    release_service = service(completed_run())
    release_service.record(ReleaseDecisionRequest(**request()))
    with pytest.raises(ValueError, match="already bound"):
        release_service.record(ReleaseDecisionRequest(**request(rationale="Changed after approval.")))


def test_failed_gate_requires_an_explicit_approved_exception() -> None:
    release_service = service(completed_run(passed=False))

    with pytest.raises(ValueError, match="gate"):
        release_service.record(ReleaseDecisionRequest(**request()))
    result = release_service.record(ReleaseDecisionRequest(**request(
        exception_approved=True, exception_rationale="Sponsor accepted the documented risk.",
    )))
    assert result.exception_approved is True


def test_approval_rechecks_active_generation_and_gold_subset_at_commit_time() -> None:
    stale = EvidenceStore(EvaluationReportEligibility(
        generation_id="dataset-v1", eligibility="exploratory", approved_gold_count=40,
        is_current_generation=False, is_superseded=True,
    ))
    with pytest.raises(ValueError, match="superseded"):
        service(completed_run(), stale).record(ReleaseDecisionRequest(**request()))

    too_many_run = completed_run(item_count=51)
    too_many_gold = EvidenceStore(EvaluationReportEligibility(
        generation_id="dataset-v1", eligibility="exploratory", approved_gold_count=51,
        is_current_generation=True, is_superseded=False,
    ))
    with pytest.raises(ValueError, match="40–50"):
        service(too_many_run, too_many_gold).record(ReleaseDecisionRequest(**request()))

    too_few_run = completed_run(item_count=39)
    too_few_gold = EvidenceStore(EvaluationReportEligibility(
        generation_id="dataset-v1", eligibility="exploratory", approved_gold_count=39,
        is_current_generation=True, is_superseded=False,
    ))
    with pytest.raises(ValueError, match="40–50"):
        service(too_few_run, too_few_gold).record(ReleaseDecisionRequest(**request()))


def test_dataset_revision_edit_invalidates_release_decision_without_mutating_run() -> None:
    run = completed_run()
    evidence = EvidenceStore()
    release_service = service(run, evidence)
    original_snapshot = evidence.run_snapshots[run.run_id][0]
    evidence.heads[run.corpus_version_id] = evidence.heads[run.corpus_version_id].model_copy(
        update={"revision": 8, "dataset_sha256": "9" * 64}
    )

    with pytest.raises(ValueError, match="current dataset revision"):
        release_service.record(ReleaseDecisionRequest(**request()))

    assert evidence.run_snapshots[run.run_id][0] == original_snapshot
    assert release_service.decisions.get("release-1") is None


def test_release_requires_twenty_approved_items_in_each_outcome_group() -> None:
    run = completed_run(item_count=40)
    evidence = EvidenceStore()
    release_service = service(run, evidence)
    evidence.attach_run(run, answerable_count=19)

    with pytest.raises(ValueError, match="20 answerable and 20 refusal/out-of-domain"):
        release_service.record(ReleaseDecisionRequest(**request()))


def test_approval_requires_exactly_the_six_release_metrics() -> None:
    incomplete = completed_run().model_copy(update={"metrics": completed_run().metrics[:-1]})
    with pytest.raises(ValueError, match="six required"):
        service(incomplete).record(ReleaseDecisionRequest(**request()))


def test_approval_requires_a_complete_attestation_bound_to_the_run() -> None:
    run = completed_run()
    with pytest.raises(ValueError, match="completed batch attestation"):
        service(run, batch_reviews=AttestedReviewStore(run, decision="rejected")).record(
            ReleaseDecisionRequest(**request())
        )
    with pytest.raises(ValueError, match="batch attestation is stale"):
        service(run, batch_reviews=AttestedReviewStore(run, valid=False)).record(
            ReleaseDecisionRequest(**request())
        )


def test_approval_rejects_legacy_unbound_batch_attestation() -> None:
    run = completed_run()
    with pytest.raises(ValueError, match="unbound"):
        service(run, batch_reviews=AttestedReviewStore(
            run, semantic_assessment_binding="legacy_unbound",
        )).record(ReleaseDecisionRequest(**request()))


def test_release_accepts_refusal_exception_only_with_scope_and_positive_exact_assessment() -> None:
    run = completed_run()
    item_id = run.selected_item_ids[-1]
    batch = AttestedReviewStore(run)
    exception = DurableExceptionRecord(
        generation_id=run.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=batch.review.summary.candidate_snapshot_sha256,
        corpus_version_id=run.corpus_version_id,
        sample_identity="refusal-exception",
        item_id=item_id,
        candidate_chunk_ids=[],
        content_hashes=[],
        provenance_digests=[],
        status="exception",
        reason="explicit_unanswerable",
        expected_refusal=True,
        decision_version="batch-review-v1",
    )
    batch.review = BatchReview(
        summary=batch.review.summary.model_copy(update={
            "status": "exception",
            "eligible_count": len(run.selected_item_ids) - 1,
            "exception_count": 1,
        }),
        samples=batch.review.samples,
        exceptions=(exception,),
    )
    refusal_decision = DurableExceptionDecision(
        generation_id=run.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=batch.review.summary.candidate_snapshot_sha256,
        corpus_version_id=run.corpus_version_id,
        sample_identity=exception.sample_identity,
        item_id=item_id,
        decision="attested",
        actor="admin",
        decided_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        decision_version="batch-review-v1",
        expected_refusal=True,
        corpus_scope_confirmed=True,
    )
    refusal_assessment = DurableSemanticAssessmentReference(
        item_id=item_id,
        assessment_id=f"assessment-{item_id}",
        output_digest="a" * 64,
        scorer_version="scorer-v1",
        reviewer_actor="reviewer",
        reviewed_at=datetime(2026, 11, 5, tzinfo=timezone.utc),
        refusal_judgment=True,
    )
    batch.result = batch.result.model_copy(update={
        "exception_decisions": [refusal_decision],
        "semantic_assessments": [*batch.result.semantic_assessments, refusal_assessment],
        "refusal_confirmation_item_ids": [item_id],
    })

    assert service(run, batch_reviews=batch).record(ReleaseDecisionRequest(**request()))

    batch.result = batch.result.model_copy(update={
        "semantic_assessments": [
            *batch.result.semantic_assessments[:-1],
            refusal_assessment.model_copy(update={"refusal_judgment": False}),
        ],
    })
    with pytest.raises(ValueError, match="stale, incomplete, or unbound"):
        service(run, batch_reviews=batch).record(ReleaseDecisionRequest(**request()))


def test_approval_requires_explicit_decisions_for_every_batch_exception() -> None:
    run = completed_run()
    batch = AttestedReviewStore(run)
    first = run.selected_item_ids[-1]
    exception = DurableExceptionRecord(
        generation_id=run.dataset_version_id,
        run_id=run.run_id,
        candidate_snapshot_sha256=batch.review.summary.candidate_snapshot_sha256,
        corpus_version_id=run.corpus_version_id,
        sample_identity="exception-1",
        item_id=first,
        candidate_chunk_ids=[],
        content_hashes=[],
        provenance_digests=[],
        status="exception",
        reason="unresolved_support",
        decision_version="batch-review-v1",
    )
    batch.review = BatchReview(
        summary=batch.review.summary.model_copy(update={
            "status": "exception", "eligible_count": len(run.selected_item_ids) - 1, "exception_count": 1,
        }),
        samples=batch.review.samples,
        exceptions=(exception,),
    )
    with pytest.raises(ValueError, match="batch attestation is stale, incomplete, or unbound"):
        service(run, batch_reviews=batch).record(ReleaseDecisionRequest(**request()))


def test_approval_fails_closed_when_batch_review_repository_is_missing() -> None:
    with pytest.raises(ValueError, match="batch-attestation repository"):
        service(completed_run(), batch_reviews=None).record(ReleaseDecisionRequest(**request()))


def test_approval_fails_closed_when_reviewed_output_digest_changes() -> None:
    run = completed_run()
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(run)
    evidence = EvidenceStore()
    evidence.attach_run(run)
    release_service = ReleaseDecisionService(
        runs, InMemoryReleaseDecisionStore(), evidence,
        support_evidence=SupportEvidenceStore(run),
        batch_reviews=AttestedReviewStore(run),
        stages=StageStore(run, stale_output=True),
    )
    with pytest.raises(ValueError, match="stale exact-output"):
        release_service.record(ReleaseDecisionRequest(**request()))


def test_release_api_requires_admin_and_persists_decision() -> None:
    admin = UserProfile(user_id="admin", email="admin@example.com", username="admin",
                        approval_state="approved", role="admin", email_verified=True)
    app.dependency_overrides[get_admin_user] = lambda: admin
    app.dependency_overrides[release_decision_service] = lambda: service(completed_run())
    try:
        response = TestClient(app).post("/api/releases", json=request())
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["gate_report_ids"] == ["gate-report-1"]
    finally:
        app.dependency_overrides.clear()


def test_release_api_rejects_non_admin() -> None:
    app.dependency_overrides[get_admin_user] = lambda: (_ for _ in ()).throw(
        IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)
    )
    try:
        response = TestClient(app).post("/api/releases", json=request())
        assert response.status_code == status.HTTP_403_FORBIDDEN
    finally:
        app.dependency_overrides.clear()
