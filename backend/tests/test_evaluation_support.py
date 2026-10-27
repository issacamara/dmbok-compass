from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pytest

from app.contracts import (
    CorpusServingBinding,
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItem,
    EvaluationRun,
    GoldAnnotation,
)
from app.evaluation.job import EvaluationOutcome, EvaluationWorker, InMemoryEvaluationRunStore
from app.evaluation.support import (
    InMemorySupportEvidenceStore,
    StoredSupportChunk,
    SupportPreparationError,
    prepare_support,
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


def chunk(chunk_id: str, text: str, *, ordinal: int = 0) -> StoredSupportChunk:
    digest = sha256(text.encode()).hexdigest()
    from app.contracts import DocumentChunk

    return StoredSupportChunk(
        DocumentChunk(
            chunk_id=chunk_id,
            corpus_version_id="corpus-v1",
            page=ordinal + 1,
            section="Section",
            chunk_ordinal=ordinal,
            content_hash=digest,
            text=text,
            embedding=[0.0] * 768,
        ),
        chunk_id,
        "d" * 64,
    )


@dataclass
class Source:
    records: tuple[StoredSupportChunk, ...]
    provenance: str = "d" * 64

    def chunks(self, *, corpus_version_id: str) -> tuple[StoredSupportChunk, ...]:
        assert corpus_version_id == "corpus-v1"
        return self.records

    def source_provenance_sha256(self, *, corpus_version_id: str) -> str:
        assert corpus_version_id == "corpus-v1"
        return self.provenance


def run(item_ids: list[str]) -> EvaluationRun:
    return EvaluationRun(
        run_id="run-v1",
        dataset_version_id="dataset-v1",
        corpus_version_id="corpus-v1",
        status="queued",
        selected_item_ids=item_ids,
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        candidate_snapshot=snapshot(),
    )


def test_unique_normalized_match_is_auto_approved_and_retry_is_idempotent() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1",
            expected_answer_or_rubric="A",
            review_status="approved",
            supporting_passage="the   exact passage",
        )],
    )
    store = InMemorySupportEvidenceStore()
    source = Source((chunk("chunk-1", "Before THE exact\npassage after"),))
    first = prepare_support(run(["item-1"]), dataset, source, store)
    second = prepare_support(run(["item-1"]), dataset, source, store)

    assert first.evidence[0].match_status == "auto_approved"
    assert first.evidence[0].selected_chunk is not None
    assert first.marker == second.marker
    assert first.evidence == second.evidence


def test_duplicate_missing_boundary_and_unanswerable_items_remain_unresolved() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id=f"item-{i}", question="Q", dataset_version_id="dataset-v1") for i in range(4)],
        gold_annotations=[
            GoldAnnotation(
                item_id="item-0", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="duplicate",
            ),
            GoldAnnotation(
                item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="absent",
            ),
            GoldAnnotation(
                item_id="item-2", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="left right",
            ),
            GoldAnnotation(
                item_id="item-3", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="ignored",
                semantic_assessment={
                    "response_sha256": "e" * 64, "scorer_version": "v1", "expected_refusal": True,
                },
            ),
        ],
    )
    result = prepare_support(
        run([f"item-{i}" for i in range(4)]),
        dataset,
        Source((
            chunk("a", "duplicate"), chunk("b", "duplicate", ordinal=1),
            chunk("c", "left", ordinal=2), chunk("d", "right", ordinal=3),
        )),
        InMemorySupportEvidenceStore(),
    )
    by_id = {item.item_id: item for item in result.evidence}

    assert by_id["item-0"].match_status == "ambiguous"
    assert by_id["item-0"].candidate_chunk_ids == ["a", "b"]
    assert by_id["item-1"].reason == "missing_supporting_passage"
    assert by_id["item-2"].reason == "support_spans_chunk_boundary"
    assert by_id["item-2"].candidate_chunk_ids == ["c", "d"]
    assert by_id["item-3"].reason == "explicit_unanswerable"
    assert by_id["item-3"].expected_refusal is True
    assert result.marker.approved_item_count == 0


def test_invalid_chunk_metadata_is_unresolved_and_interrupted_retry_has_no_marker() -> None:
    invalid = chunk("bad", "support")
    assert invalid.chunk is not None
    invalid = StoredSupportChunk(
        invalid.chunk.model_copy(update={"content_hash": "f" * 64}),
        "bad",
        "d" * 64,
    )
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            relevant_chunk_ids=["bad"], supporting_passage="support",
        )],
    )
    result = prepare_support(run(["item-1"]), dataset, Source((invalid,)), InMemorySupportEvidenceStore())
    assert result.evidence[0].match_status == "invalid"
    assert result.evidence[0].reason == "invalid_chunk_metadata"

    class FailingStore(InMemorySupportEvidenceStore):
        def create_evidence(self, evidence):  # type: ignore[no-untyped-def]
            raise SupportPreparationError("simulated interruption")

    store = FailingStore()
    with pytest.raises(SupportPreparationError, match="interruption"):
        prepare_support(run(["item-1"]), dataset, Source((chunk("good", "support"),)), store)
    assert store.get_marker(run_id="run-v1") is None


def test_worker_prepares_support_before_scoring() -> None:
    run_record = run(["item-1"])
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(run_record)
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
    )
    events: list[str] = []

    result = EvaluationWorker(runs).execute(
        run_record.run_id,
        dataset,
        lambda _: (events.append("score"), EvaluationOutcome(True, True, True, True, True, 1))[1],
        support_preparer=lambda _run, _dataset: events.append("prepare"),
    )

    assert result.status == "completed"
    assert events == ["prepare", "score"]


def test_stale_run_snapshot_is_rejected_before_any_evidence_is_written() -> None:
    stale = run(["item-1"]).model_copy(update={"corpus_version_id": "other-corpus"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
    )
    with pytest.raises(SupportPreparationError, match="not bound"):
        prepare_support(stale, dataset, Source(()), InMemorySupportEvidenceStore())
