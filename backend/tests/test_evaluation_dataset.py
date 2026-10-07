from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.contracts import EvaluationDataset, EvaluationItem, GoldAnnotation
from app.evaluation.dataset import (
    EvaluationDatasetError,
    GoldSubsetError,
    InMemoryEvaluationDatasetStore,
    ReviewTransitionError,
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
