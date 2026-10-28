from __future__ import annotations

from hashlib import sha256

from app.contracts import DocumentChunk, EvaluationDataset, EvaluationItem, GoldAnnotation
from app.evaluation.job import EvaluationOutcome, EvaluationWorker
from app.evaluation.support import InMemorySupportEvidenceStore, StoredSupportChunk, prepare_support
from test_evaluation_generations import FakeFirestore, run_fake_transaction
from test_evaluation_support import Source, chunk, run
from test_integrated_release_journey import _Snapshots, _dataset, _snapshot
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.job import (
    EvaluationJobService,
    FirestoreEvaluationRunStore,
    InMemoryEvaluationJobDispatcher,
)


def test_a23_representative_cases_remain_reviewable_without_auto_approval() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-a23",
        items=[EvaluationItem(item_id=f"item-{index}", question="Q", dataset_version_id="dataset-a23") for index in range(7)],
        gold_annotations=[
            GoldAnnotation(item_id="item-0", expected_answer_or_rubric="A", review_status="approved", supporting_passage="unique support"),
            GoldAnnotation(item_id="item-1", expected_answer_or_rubric="A", review_status="approved", supporting_passage="duplicate"),
            GoldAnnotation(item_id="item-2", expected_answer_or_rubric="A", review_status="approved", supporting_passage="missing"),
            GoldAnnotation(item_id="item-3", expected_answer_or_rubric="A", review_status="approved", supporting_passage="invalid hash"),
            GoldAnnotation(item_id="item-4", expected_answer_or_rubric="A", review_status="approved", supporting_passage="short"),
            GoldAnnotation(item_id="item-5", expected_answer_or_rubric="A", review_status="approved", supporting_passage="left right"),
            GoldAnnotation(item_id="item-6", expected_answer_or_rubric="A", review_status="approved", supporting_passage="ignored", semantic_assessment={"response_sha256": "a" * 64, "scorer_version": "v1", "expected_refusal": True}),
        ],
    )
    invalid = chunk("invalid", "invalid hash")
    assert invalid.chunk is not None
    invalid = StoredSupportChunk(invalid.chunk.model_copy(update={"content_hash": "f" * 64}), "invalid", "d" * 64)
    result = prepare_support(
        run([f"item-{index}" for index in range(7)]).model_copy(update={"dataset_version_id": "dataset-a23"}),
        dataset,
        Source((
            chunk("unique", "Unique support"),
            chunk("duplicate-a", "duplicate"), chunk("duplicate-b", "duplicate", ordinal=1),
            invalid,
            chunk("short", "short", ordinal=3),
            chunk("left", "left", ordinal=4), chunk("right", "right", ordinal=5),
        )),
        InMemorySupportEvidenceStore(),
    )
    statuses = {item.item_id: item for item in result.evidence}
    assert statuses["item-0"].match_status == "auto_approved"
    assert statuses["item-1"].match_status == "ambiguous"
    assert statuses["item-2"].reason == "missing_supporting_passage"
    assert statuses["item-3"].match_status == "invalid"
    assert statuses["item-4"].match_status == "auto_approved"
    assert statuses["item-5"].reason == "support_spans_chunk_boundary"
    assert statuses["item-6"].reason == "explicit_unanswerable"


def test_integrated_run_requires_run_bound_support_and_separate_exact_output_review() -> None:
    client = FakeFirestore()
    generations = FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)
    dataset, raw_chunks = _dataset("generation-v22")
    snapshot = _snapshot("corpus-v1")
    review = remap_gold_annotations(
        dataset,
        raw_chunks,
        CandidateGoldBinding("corpus-v1", sha256(snapshot.model_dump_json().encode()).hexdigest(), "e" * 64),
    )
    generations.stage(dataset, candidate_review=review)
    generations.activate("generation-v22", expected_active_generation_id=None)
    for item in dataset.items:
        generations.review("generation-v22", item.item_id, "approved")

    support_store = InMemorySupportEvidenceStore()
    source = Source(tuple(
        StoredSupportChunk(chunk, chunk.chunk_id, "d" * 64)
        for chunk in (
            DocumentChunk(
                chunk_id=f"candidate-{item.item_id}", corpus_version_id="corpus-v1", page=index + 1,
                section="Candidate section", chunk_ordinal=0,
                content_hash=sha256(f"Candidate support record [{index:02d}]".encode()).hexdigest(),
                text=f"Candidate support record [{index:02d}]", embedding=[0.0] * 768,
            )
            for index, item in enumerate(dataset.items)
        )
    ))
    runs = FirestoreEvaluationRunStore(client)
    jobs = EvaluationJobService(
        generations, runs, InMemoryEvaluationJobDispatcher(), _Snapshots(), support_evidence=support_store,
    )
    run_record = jobs.launch(corpus_version_id="corpus-v1", configuration_version_id="config-v1", model_version_id="model-v1", item_ids=None)
    assert jobs.status(run_record.run_id).report_eligibility.eligibility == "exploratory"

    worker = EvaluationWorker(runs)
    pending = worker.execute(
        run_record.run_id,
        dataset,
        lambda item_id: EvaluationOutcome(
            True, True, True, True, True, 100,
            assessment_status="awaiting_review", output_handle=f"output-{item_id}", output_digest=sha256(item_id.encode()).hexdigest(),
        ),
        support_preparer=lambda pinned_run, pinned_dataset: prepare_support(pinned_run, pinned_dataset, source, support_store),
        support_evidence=support_store.get_evidence,
    )
    assert pending.metrics == []
    assert jobs.status(run_record.run_id).report_eligibility.approved_gold_count == 30
    assert jobs.status(run_record.run_id).report_eligibility.eligibility == "release_evidence"

    for item in dataset.items:
        digest = sha256(item.item_id.encode()).hexdigest()
        worker.review_item(run_record.run_id, item.item_id, output_digest=digest, assessment=EvaluationOutcome(True, True, True, True, True, 100))
    finalized = worker.finalize(run_record.run_id)
    assert finalized.status == "completed"
    assert all(metric.passed for metric in finalized.metrics)
    restarted = FirestoreEvaluationRunStore(client)
    assert restarted.get(run_record.run_id) == finalized
