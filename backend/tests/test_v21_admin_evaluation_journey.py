from __future__ import annotations

from hashlib import sha256

import pytest

from app.contracts import CandidateIdentity, RetrievalResult
from app.evaluation.assessments import response_fingerprint, score_assessment
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
from app.evaluation.job import EvaluationOutcome, EvaluationWorker
from test_evaluation_generations import FakeFirestore, run_fake_transaction
from test_integrated_release_journey import _Snapshots, _dataset, _snapshot
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.job import (
    EvaluationJobService,
    FirestoreEvaluationRunStore,
    InMemoryEvaluationJobDispatcher,
)


def _retrieval(chunk_id: str = "candidate-item-0") -> RetrievalResult:
    return RetrievalResult(
        candidate=CandidateIdentity(
            artifact_id="candidate-image@sha256:" + "d" * 64,
            installed_lock_sha256="c" * 64,
            corpus_version_id="corpus-v1",
            configuration_version_id="config-v1",
            scorer_version="evaluation-gates-v1",
        ),
        ordered_chunk_ids=[chunk_id],
        scores=[0.91],
        score_semantics="cosine_similarity",
        strategy="dense",
        stage_status="succeeded",
        search_ready=True,
    )


def test_final_journey_requires_current_binding_and_exact_output_review() -> None:
    """Compose the real generation, run, worker, and review repositories."""

    client = FakeFirestore()
    generations = FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)
    dataset, chunks = _dataset("generation-v1")
    snapshot = _snapshot("corpus-v1")
    candidate_review = remap_gold_annotations(
        dataset,
        chunks,
        CandidateGoldBinding(
            corpus_version_id="corpus-v1",
            candidate_snapshot_sha256=sha256(snapshot.model_dump_json().encode()).hexdigest(),
            source_provenance_sha256="e" * 64,
        ),
    )

    generations.stage(dataset, candidate_review=candidate_review)
    generations.activate("generation-v1", expected_active_generation_id=None)
    for item in dataset.items:
        generations.review("generation-v1", item.item_id, "approved")

    runs = FirestoreEvaluationRunStore(client)
    jobs = EvaluationJobService(generations, runs, InMemoryEvaluationJobDispatcher(), _Snapshots())
    run = jobs.launch(
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        item_ids=None,
    )
    worker = EvaluationWorker(runs)
    pending = worker.execute(
        run.run_id,
        dataset,
        lambda item_id: EvaluationOutcome(
            retrieval_success=True,
            grounded=True,
            citation_correct=True,
            answer_quality=True,
            refusal_correct=True,
            response_time_ms=1_000,
            retrieval=_retrieval(f"candidate-{item_id}"),
            generation_outcome="answer",
            output_handle=f"attempt-{item_id}",
            output_digest=sha256(f"answer-{item_id}".encode()).hexdigest(),
            assessment_status="awaiting_review",
        ),
    )

    assert pending.status == "completed"
    assert pending.metrics == []
    assert worker.finalize(run.run_id).metrics == []

    with pytest.raises(ValueError, match="digest"):
        worker.review_item(
            run.run_id,
            "item-0",
            output_digest="0" * 64,
            assessment=EvaluationOutcome(
                retrieval_success=True,
                grounded=True,
                citation_correct=True,
                answer_quality=True,
                refusal_correct=True,
                response_time_ms=1_000,
            ),
        )

    for item in dataset.items:
        digest = sha256(f"answer-{item.item_id}".encode()).hexdigest()
        worker.review_item(
            run.run_id,
            item.item_id,
            output_digest=digest,
            assessment=EvaluationOutcome(
                retrieval_success=True,
                grounded=True,
                citation_correct=True,
                answer_quality=True,
                refusal_correct=True,
                response_time_ms=1_000,
            ),
        )
    finalized = worker.finalize(run.run_id)
    assert finalized.metrics
    assert all(metric.passed for metric in finalized.metrics)


def test_retrieval_hit_is_independent_from_changed_answer_and_operational_failure() -> None:
    reviewed = {
        "response_sha256": response_fingerprint(
            outcome="answer", answer_text="Original answer", cited_chunk_ids={"chunk-1"}
        ),
        "scorer_version": "evaluation-gates-v1",
        "claim_support": True,
        "claim_citation_support": True,
        "rubric_passed": True,
        "expected_refusal": False,
    }
    from app.contracts import GoldAnnotation, ReviewedSemanticAssessment

    annotation = GoldAnnotation(
        item_id="item-1",
        expected_answer_or_rubric="Reviewed answer",
        relevant_chunk_ids=["chunk-1"],
        review_status="approved",
        semantic_assessment=ReviewedSemanticAssessment(**reviewed),
    )
    changed = score_assessment(
        annotation,
        outcome="answer",
        answer_text="Changed answer text",
        retrieved_chunk_ids={"chunk-1"},
        cited_chunk_ids={"chunk-1"},
        scorer_version="evaluation-gates-v1",
    )
    outage = score_assessment(
        annotation,
        outcome="operational_error",
        answer_text=None,
        retrieved_chunk_ids={"chunk-1"},
        cited_chunk_ids=set(),
        scorer_version="evaluation-gates-v1",
        operational_error=True,
    )

    assert changed.retrieval_success is True
    assert changed.answer_quality is False
    assert outage.retrieval_success is True
    assert outage.refusal_correct is False
