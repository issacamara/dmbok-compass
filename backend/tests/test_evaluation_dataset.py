from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.contracts import DocumentChunk, EvaluationDataset, EvaluationItem, GoldAnnotation
from app.evaluation.dataset import (
    CandidateBindingError,
    CandidateChunk,
    CandidateGoldBinding,
    EvaluationDatasetError,
    GoldSubsetError,
    InMemoryEvaluationDatasetStore,
    ReviewTransitionError,
    remap_gold_annotations,
)


def make_dataset(annotation_count: int = 1) -> EvaluationDataset:
    items = [
        EvaluationItem(
            item_id=f"item-{index}",
            question=f"What is question {index}?",
            dataset_version_id="dataset-v1",
            category=("definitions", "explanations", "comparisons", "study", "scenarios")[index % 5],
        )
        for index in range(max(annotation_count, 1))
    ]
    annotations = [
        GoldAnnotation(
            item_id=f"item-{index}",
            expected_answer_or_rubric="The evaluator-approved rubric.",
            relevant_chunk_ids=[f"chunk-{index}"],
            review_status="candidate",
        )
        for index in range(annotation_count)
    ]
    return EvaluationDataset(dataset_version_id="dataset-v1", items=items, gold_annotations=annotations)


def test_evaluation_schema_has_only_supported_question_categories() -> None:
    assert {item.category for item in make_dataset(5).items} == {
        "definitions", "explanations", "comparisons", "study", "scenarios"
    }
    with pytest.raises(ValidationError):
        EvaluationItem(item_id="item", question="Question", dataset_version_id="v1", category="chat")  # type: ignore[arg-type]


def test_store_rejects_mixed_dataset_versions_and_unknown_annotations() -> None:
    store = InMemoryEvaluationDatasetStore()
    dataset = make_dataset()
    dataset.items[0].dataset_version_id = "other"
    with pytest.raises(EvaluationDatasetError, match="belong to the dataset version"):
        store.create(dataset)

    store.create(make_dataset())
    with pytest.raises(EvaluationDatasetError, match="unknown annotation"):
        store.review_annotation("dataset-v1", "missing", "approved")


def test_review_transitions_are_explicit_and_terminal() -> None:
    store = InMemoryEvaluationDatasetStore()
    store.create(make_dataset())
    approved = store.review_annotation("dataset-v1", "item-0", "approved")
    assert approved.gold_annotations[0].review_status == "approved"
    with pytest.raises(ReviewTransitionError):
        store.review_annotation("dataset-v1", "item-0", "rejected")


def test_dataset_versions_are_immutable_and_idempotent() -> None:
    store = InMemoryEvaluationDatasetStore()
    dataset = make_dataset()
    assert store.create(dataset) == dataset
    assert store.create(dataset) == dataset
    changed = dataset.model_copy(update={"items": [dataset.items[0].model_copy(update={"question": "Changed"})]})
    with pytest.raises(EvaluationDatasetError, match="already bound"):
        store.create(changed)


def test_gold_subset_requires_30_to_50_human_approvals() -> None:
    store = InMemoryEvaluationDatasetStore()
    store.create(make_dataset(29))
    for index in range(29):
        store.review_annotation("dataset-v1", f"item-{index}", "approved")
    with pytest.raises(GoldSubsetError):
        store.approve_gold_subset("dataset-v1")

    store = InMemoryEvaluationDatasetStore()
    store.create(make_dataset(30))
    for index in range(30):
        store.review_annotation("dataset-v1", f"item-{index}", "approved")
    store.approve_gold_subset("dataset-v1")
    assert store.is_gold_subset_approved("dataset-v1")


def _candidate_chunk(
    chunk_id: str,
    text: str,
    *,
    page: int,
    section: str,
    section_origin: str = "explicit",
) -> CandidateChunk:
    return CandidateChunk(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            corpus_version_id="candidate-corpus",
            page=page,
            section=section,
            chunk_ordinal=0,
            content_hash="a" * 64,
            text=text,
            embedding=[0.0] * 768,
        ),
        section_origin=section_origin,  # type: ignore[arg-type]
    )


def _candidate_binding() -> CandidateGoldBinding:
    return CandidateGoldBinding(
        corpus_version_id="candidate-corpus",
        candidate_snapshot_sha256="b" * 64,
        source_provenance_sha256="c" * 64,
    )


def test_remap_uses_candidate_provenance_and_never_carries_approval_forward() -> None:
    categories = ("definitions", "explanations", "comparisons", "study", "scenarios")
    supports = ("definition support", "carried support", "unclassified support", "ambiguous support", "missing support")
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v2",
        items=[
            EvaluationItem(
                item_id=f"item-{index}",
                question=f"Question {index}",
                dataset_version_id="dataset-v2",
                category=category,
            )
            for index, category in enumerate(categories)
        ],
        gold_annotations=[
            GoldAnnotation(
                item_id=f"item-{index}",
                expected_answer_or_rubric=f"Oracle outcome: {('answerable', 'partial', 'unanswerable')[index % 3]}",
                relevant_chunk_ids=[f"legacy-{index}"],
                review_status="candidate",
                supporting_passage=support,
            )
            for index, support in enumerate(supports)
        ],
    )
    review = remap_gold_annotations(
        dataset,
        [
            _candidate_chunk("chunk-definition", "A definition support appears here.", page=1, section="Definitions"),
            _candidate_chunk("chunk-carried", "A carried support appears here.", page=2, section="Context", section_origin="carried"),
            _candidate_chunk("chunk-unclassified", "An unclassified support appears here.", page=3, section="Unclassified"),
            _candidate_chunk("chunk-ambiguous-a", "An ambiguous support appears here.", page=4, section="A"),
            _candidate_chunk("chunk-ambiguous-b", "An ambiguous support appears here.", page=5, section="B"),
            _candidate_chunk("legacy-4", "The prior supporting text was changed.", page=6, section="Changed"),
        ],
        _candidate_binding(),
    )

    by_item = {remap.item_id: remap for remap in review.remaps}
    assert by_item["item-0"].remap_status == "matched"
    assert by_item["item-0"].relevant_chunk_ids == ("chunk-definition",)
    assert (by_item["item-0"].page, by_item["item-0"].section) == (1, "Definitions")
    assert "review_carried_heading" in by_item["item-1"].review_reasons
    assert "review_unclassified_section" in by_item["item-2"].review_reasons
    assert by_item["item-3"].remap_status == "ambiguous"
    assert by_item["item-4"].remap_status == "changed"
    assert all(remap.model_dump()["review_status"] == "candidate" for remap in review.remaps)


def test_remap_rejects_chunks_from_another_corpus() -> None:
    chunk = _candidate_chunk("chunk-1", "support", page=1, section="Section")
    wrong_chunk = CandidateChunk(chunk=chunk.chunk.model_copy(update={"corpus_version_id": "legacy-corpus"}))
    with pytest.raises(CandidateBindingError, match="bound candidate corpus"):
        remap_gold_annotations(make_dataset(), [wrong_chunk], _candidate_binding())


def test_remap_flags_missing_and_unknown_support_for_review() -> None:
    missing = make_dataset().model_copy(
        update={"gold_annotations": [make_dataset().gold_annotations[0].model_copy(update={"supporting_passage": None})]}
    )
    unknown = make_dataset().model_copy(
        update={"gold_annotations": [make_dataset().gold_annotations[0].model_copy(update={"supporting_passage": "Unknown support"})]}
    )
    chunk = _candidate_chunk("candidate", "Different candidate text", page=1, section="Section")

    assert remap_gold_annotations(missing, [chunk], _candidate_binding()).remaps[0].remap_status == "missing"
    assert remap_gold_annotations(unknown, [chunk], _candidate_binding()).remaps[0].remap_status == "unknown"
