from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest

from app.contracts import (
    BatchAttestationRequest,
    CorpusServingBinding,
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItem,
    EvaluationRun,
    GoldAnnotation,
    RunBoundSupportEvidence,
    VerifiedChunkChoice,
)
from app.evaluation.batch_reviews import (
    BatchReviewConflictError,
    BatchReviewStaleError,
    FirestoreBatchReviewRepository,
    InMemoryBatchReviewRepository,
    NoEligibleCandidatesError,
    create_batch_review,
)


def snapshot() -> EvaluationCandidateSnapshot:
    return EvaluationCandidateSnapshot(
        corpus_binding=CorpusServingBinding(
            corpus_version_id="corpus-v1",
            vector_collection="document_chunks_v2",
            vector_schema_version="2",
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
        ),
        project="project",
        location="europe-west1",
        answer_provider="openrouter",
        primary_model="primary",
        fallback_model=None,
        json_schema_supported=True,
        request_timeout_ms=1000,
        generation_timeout_ms=1000,
        prompt_version="prompt-v1",
        retrieval_policy_version="retrieval-v1",
        evidence_policy_version="evidence-v1",
        scorer_version="scorer-v1",
        dependency_lock_sha256="c" * 64,
        integration_artifact="artifact",
        runtime_version="revision-v1",
    )


def run(item_ids: list[str]) -> EvaluationRun:
    return EvaluationRun(
        run_id="run-v1",
        dataset_version_id="generation-v1",
        corpus_version_id="corpus-v1",
        status="completed",
        selected_item_ids=item_ids,
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        candidate_snapshot=snapshot(),
    )


def evidence(item_id: str, *, status: str = "eligible", category: str = "definitions") -> RunBoundSupportEvidence:
    del category  # Category is intentionally owned by the dataset item.
    choice = VerifiedChunkChoice(
        chunk_id=f"chunk-{item_id}",
        corpus_version_id="corpus-v1",
        content_hash=sha256(item_id.encode()).hexdigest(),
        page=1,
        section="Section",
        source_provenance_sha256="d" * 64,
    )
    return RunBoundSupportEvidence(
        run_id="run-v1",
        item_id=item_id,
        candidate_snapshot_sha256=sha256(snapshot().model_dump_json().encode()).hexdigest(),
        corpus_version_id="corpus-v1",
        source_provenance_sha256="d" * 64,
        match_status=status,
        candidate_chunk_ids=[choice.chunk_id] if status == "eligible" else [],
        verified_chunks=[choice] if status == "eligible" else [],
        selected_chunk=choice if status == "eligible" else None,
        reason=None if status == "eligible" else f"{status}-support",
    )


def dataset(item_ids: list[str]) -> EvaluationDataset:
    return EvaluationDataset(
        dataset_version_id="generation-v1",
        items=[
            EvaluationItem(
                item_id=item_id,
                question=f"Question {item_id}",
                dataset_version_id="generation-v1",
                category=("comparisons" if item_id.endswith("-comparison") else "definitions"),
            )
            for item_id in item_ids
        ],
        gold_annotations=[
            GoldAnnotation(item_id=item_id, expected_answer_or_rubric="Rubric", review_status="approved")
            for item_id in item_ids
        ],
    )


def test_sample_is_deterministic_bounded_and_category_aware() -> None:
    item_ids = [f"item-{index}" for index in range(50)] + ["item-comparison"]
    evaluation_run = run(item_ids)
    records = [evidence(item_id) for item_id in item_ids]

    first = create_batch_review(evaluation_run, dataset(item_ids), records)
    second = create_batch_review(evaluation_run, dataset(item_ids), list(reversed(records)))

    assert first.summary == second.summary
    assert first.samples == second.samples
    assert len(first.samples) == 10
    assert "item-comparison" in first.summary.sampled_item_ids
    assert all("question" not in sample.model_dump() for sample in first.samples)


def test_all_exceptions_are_persisted_and_block_implicit_attestation() -> None:
    item_ids = ["item-eligible", "item-missing", "item-ambiguous"]
    evaluation_run = run(item_ids)
    review = create_batch_review(
        evaluation_run,
        dataset(item_ids),
        [evidence("item-eligible"), evidence("item-missing", status="missing"), evidence("item-ambiguous", status="ambiguous")],
    )
    assert review.summary.status == "exception"
    assert {record.item_id for record in review.exceptions} == {"item-missing", "item-ambiguous"}

    repository = InMemoryBatchReviewRepository()
    repository.create_or_get(review)
    with pytest.raises(BatchReviewConflictError, match="all and only"):
        repository.attest(
            review,
            BatchAttestationRequest(decision="attested"),
            actor="admin",
            decided_at=datetime.now(UTC),
        )


def test_attestation_is_compare_and_set_and_repeated_decision_is_rejected() -> None:
    evaluation_run = run(["item-1", "item-missing"])
    review = create_batch_review(
        evaluation_run,
        dataset(["item-1", "item-missing"]),
        [evidence("item-1"), evidence("item-missing", status="missing")],
    )
    repository = InMemoryBatchReviewRepository()
    stored = repository.create_or_get(review)
    request = BatchAttestationRequest(
        decision="attested",
        exception_decisions=[{"item_id": "item-missing", "decision": "inapplicable", "reason": "Not answerable."}],
    )
    decision = repository.attest(
        stored,
        request,
        actor="admin",
        decided_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert decision.sampled_item_ids == stored.summary.sampled_item_ids
    assert decision.exception_decisions[0].item_id == "item-missing"

    with pytest.raises(BatchReviewConflictError, match="already"):
        repository.attest(stored, request, actor="admin-2", decided_at=datetime.now(UTC))

    changed = stored.summary.model_copy(update={"candidate_snapshot_sha256": "e" * 64})
    stale = stored.__class__(summary=changed, samples=stored.samples, exceptions=stored.exceptions)
    with pytest.raises(BatchReviewStaleError):
        repository.attest(stale, request, actor="admin", decided_at=datetime.now(UTC))


def test_no_eligible_candidates_is_not_a_reviewable_batch() -> None:
    with pytest.raises(NoEligibleCandidatesError, match="no eligible"):
        create_batch_review(
            run(["item-missing"]),
            dataset(["item-missing"]),
            [evidence("item-missing", status="missing")],
        )


def test_firestore_repository_round_trip_keeps_review_records_immutable() -> None:
    from test_evaluation_generations import FakeFirestore

    item_ids = ["item-1", "item-missing"]
    review = create_batch_review(
        run(item_ids),
        dataset(item_ids),
        [evidence("item-1"), evidence("item-missing", status="missing")],
    )
    client = FakeFirestore()
    repository = FirestoreBatchReviewRepository(client)
    repository.create_or_get(review)

    restarted = FirestoreBatchReviewRepository(client)
    assert restarted.get(generation_id="generation-v1", run_id="run-v1") == review
    persisted = repr(client.data)
    assert "Question item-1" not in persisted
    assert "Rubric" not in persisted
