"""Versioned evaluator-authored questions and human-reviewed annotations."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Protocol

from app.contracts import EvaluationDataset


class EvaluationDatasetError(ValueError):
    """Base error for invalid evaluation dataset operations."""


class EvaluationDatasetStore(Protocol):
    def create(self, dataset: EvaluationDataset) -> EvaluationDataset: ...
    def get(self, dataset_version_id: str) -> EvaluationDataset | None: ...


class InMemoryEvaluationDatasetStore:
    """Small deterministic store used by the worker and local tests."""

    def __init__(self, datasets: list[EvaluationDataset] | None = None) -> None:
        self._datasets = {item.dataset_version_id: item.model_copy(deep=True) for item in datasets or []}
        self._lock = RLock()

    def create(self, dataset: EvaluationDataset) -> EvaluationDataset:
        _validate_dataset(dataset)
        with self._lock:
            existing = self._datasets.get(dataset.dataset_version_id)
            if existing is not None and existing != dataset:
                raise EvaluationDatasetError("dataset_version_id is already bound to another dataset")
            self._datasets.setdefault(dataset.dataset_version_id, dataset.model_copy(deep=True))
            return deepcopy(self._datasets[dataset.dataset_version_id])

    def get(self, dataset_version_id: str) -> EvaluationDataset | None:
        with self._lock:
            dataset = self._datasets.get(dataset_version_id)
            return deepcopy(dataset) if dataset is not None else None


def _validate_dataset(dataset: EvaluationDataset) -> None:
    item_ids = [item.item_id for item in dataset.items]
    if len(item_ids) != len(set(item_ids)):
        raise EvaluationDatasetError("dataset contains duplicate item IDs")
    if {item.dataset_version_id for item in dataset.items} != {dataset.dataset_version_id}:
        raise EvaluationDatasetError("every item must belong to the dataset version")
    annotation_ids = [annotation.item_id for annotation in dataset.gold_annotations]
    if len(annotation_ids) != len(set(annotation_ids)) or not set(annotation_ids).issubset(item_ids):
        raise EvaluationDatasetError("annotations must reference unique dataset items")
