"""Versioned evaluator-authored questions and human-reviewed annotations.

Evaluation data is deliberately separate from live question handling.  The
store accepts only :class:`EvaluationDataset` records, never answer requests
or production traces, so reproducible evaluation cannot become a mechanism
for retaining user interactions.
"""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Protocol

from app.contracts import EvaluationDataset, GoldAnnotation


class EvaluationDatasetError(ValueError):
    """Base error for invalid evaluation dataset operations."""


class ReviewTransitionError(EvaluationDatasetError):
    """Raised when an annotation review transition is not allowed."""


class GoldSubsetError(EvaluationDatasetError):
    """Raised when a dataset does not have an approvable gold subset."""


class EvaluationDatasetStore(Protocol):
    def create(self, dataset: EvaluationDataset) -> EvaluationDataset: ...

    def get(self, dataset_version_id: str) -> EvaluationDataset | None: ...

    def review_annotation(
        self, dataset_version_id: str, item_id: str, review_status: str
    ) -> EvaluationDataset: ...

    def approve_gold_subset(self, dataset_version_id: str) -> EvaluationDataset: ...


class InMemoryEvaluationDatasetStore:
    """Reference store for local development and evaluation-job tests.

    Dataset versions are immutable at creation.  The only permitted later
    change is the explicit human review transition of an existing annotation.
    """

    MIN_GOLD_ITEMS = 30
    MAX_GOLD_ITEMS = 50
    _TRANSITIONS = {
        "candidate": frozenset({"approved", "rejected"}),
        "approved": frozenset(),
        "rejected": frozenset(),
    }

    def __init__(self) -> None:
        self._datasets: dict[str, EvaluationDataset] = {}
        self._gold_approved: set[str] = set()
        self._lock = RLock()

    def create(self, dataset: EvaluationDataset) -> EvaluationDataset:
        self._validate_dataset(dataset)
        with self._lock:
            existing = self._datasets.get(dataset.dataset_version_id)
            if existing is not None:
                if existing != dataset:
                    raise EvaluationDatasetError(
                        "dataset_version_id is already bound to another dataset"
                    )
                return deepcopy(existing)
            self._datasets[dataset.dataset_version_id] = dataset.model_copy(deep=True)
            return deepcopy(dataset)

    def get(self, dataset_version_id: str) -> EvaluationDataset | None:
        with self._lock:
            dataset = self._datasets.get(dataset_version_id)
            return deepcopy(dataset) if dataset is not None else None

    def review_annotation(
        self, dataset_version_id: str, item_id: str, review_status: str
    ) -> EvaluationDataset:
        if review_status not in {"candidate", "approved", "rejected"}:
            raise ReviewTransitionError(f"unknown review status: {review_status}")
        with self._lock:
            dataset = self._require(dataset_version_id)
            annotations = list(dataset.gold_annotations)
            try:
                index = next(i for i, annotation in enumerate(annotations) if annotation.item_id == item_id)
            except StopIteration as exc:
                raise EvaluationDatasetError(f"unknown annotation item: {item_id}") from exc

            current = annotations[index].review_status
            if review_status not in self._TRANSITIONS[current]:
                raise ReviewTransitionError(
                    f"cannot transition annotation {item_id} from {current} to {review_status}"
                )
            annotations[index] = annotations[index].model_copy(update={"review_status": review_status})
            updated = dataset.model_copy(update={"gold_annotations": annotations}, deep=True)
            self._datasets[dataset_version_id] = updated
            return deepcopy(updated)

    def approve_gold_subset(self, dataset_version_id: str) -> EvaluationDataset:
        with self._lock:
            dataset = self._require(dataset_version_id)
            approved = [
                annotation for annotation in dataset.gold_annotations if annotation.review_status == "approved"
            ]
            count = len(approved)
            if not self.MIN_GOLD_ITEMS <= count <= self.MAX_GOLD_ITEMS:
                raise GoldSubsetError("the approved gold subset must contain between 30 and 50 items")
            self._gold_approved.add(dataset_version_id)
            return deepcopy(dataset)

    def is_gold_subset_approved(self, dataset_version_id: str) -> bool:
        with self._lock:
            return dataset_version_id in self._gold_approved

    def _require(self, dataset_version_id: str) -> EvaluationDataset:
        dataset = self._datasets.get(dataset_version_id)
        if dataset is None:
            raise EvaluationDatasetError(f"unknown dataset version: {dataset_version_id}")
        return dataset

    @staticmethod
    def _validate_dataset(dataset: EvaluationDataset) -> None:
        item_ids = [item.item_id for item in dataset.items]
        if len(item_ids) != len(set(item_ids)):
            raise EvaluationDatasetError("dataset contains duplicate item IDs")
        item_versions = {item.dataset_version_id for item in dataset.items}
        if item_versions != {dataset.dataset_version_id}:
            raise EvaluationDatasetError("every item must belong to the dataset version")
        annotation_ids = [annotation.item_id for annotation in dataset.gold_annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise EvaluationDatasetError("dataset contains duplicate annotations")
        if not set(annotation_ids).issubset(item_ids):
            raise EvaluationDatasetError("every annotation must reference a dataset item")
