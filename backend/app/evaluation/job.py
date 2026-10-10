"""Evaluation run identity, selection, dispatch, and idempotent status storage."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from threading import RLock
from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Callable

from google.api_core.exceptions import AlreadyExists

from app.contracts import EvaluationDataset, EvaluationItemResult, EvaluationMetric, EvaluationReportEligibility, EvaluationRun
from app.evaluation.dataset import EvaluationDatasetStore


class EvaluationRunError(ValueError):
    """Raised when a run request cannot be bound to immutable inputs."""


class EvaluationRunStore(Protocol):
    def create_or_get(self, run: EvaluationRun) -> EvaluationRun: ...
    def get(self, run_id: str) -> EvaluationRun | None: ...
    def update(self, run: EvaluationRun) -> EvaluationRun: ...


class ActiveEvaluationDatasetStore(EvaluationDatasetStore, Protocol):
    """Dataset storage that can resolve the sole generation visible to launches."""

    def active_dataset(self) -> EvaluationDataset | None: ...

    def report_eligibility(self, generation_id: str) -> EvaluationReportEligibility: ...


class EvaluationJobDispatcher(Protocol):
    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None: ...


class InMemoryEvaluationRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, EvaluationRun] = {}
        self._lock = RLock()

    def create_or_get(self, run: EvaluationRun) -> EvaluationRun:
        with self._lock:
            existing = self._runs.get(run.run_id)
            if existing is not None:
                if existing.model_dump() != run.model_dump():
                    raise EvaluationRunError("run_id is already bound to another configuration")
                return deepcopy(existing)
            self._runs[run.run_id] = run.model_copy(deep=True)
            return deepcopy(run)

    def get(self, run_id: str) -> EvaluationRun | None:
        with self._lock:
            return deepcopy(self._runs.get(run_id))

    def update(self, run: EvaluationRun) -> EvaluationRun:
        with self._lock:
            if run.run_id not in self._runs:
                raise EvaluationRunError("unknown evaluation run")
            self._runs[run.run_id] = run.model_copy(deep=True)
            return deepcopy(run)


class FirestoreEvaluationRunStore:
    """Durable immutable run records, separated from evaluator-authored data."""

    collection_id = "evaluation_runs"

    def __init__(self, client: Any) -> None:
        self.runs = client.collection(self.collection_id)

    def create_or_get(self, run: EvaluationRun) -> EvaluationRun:
        reference = self.runs.document(run.run_id)
        snapshot = reference.get()
        if snapshot.exists:
            return self._same_or_conflict(snapshot, run)
        try:
            reference.create(run.model_dump(mode="json"))
            return run.model_copy(deep=True)
        except AlreadyExists:
            # A concurrent create may have won. Re-read to preserve idempotency
            # while still rejecting a hash collision with different bindings.
            snapshot = reference.get()
            if snapshot.exists:
                return self._same_or_conflict(snapshot, run)
            raise

    def get(self, run_id: str) -> EvaluationRun | None:
        snapshot = self.runs.document(run_id).get()
        return EvaluationRun.model_validate(snapshot.to_dict()) if snapshot.exists else None

    def update(self, run: EvaluationRun) -> EvaluationRun:
        reference = self.runs.document(run.run_id)
        if not reference.get().exists:
            raise EvaluationRunError("unknown evaluation run")
        reference.set(run.model_dump(mode="json"))
        return run.model_copy(deep=True)

    @staticmethod
    def _same_or_conflict(snapshot: Any, run: EvaluationRun) -> EvaluationRun:
        stored = EvaluationRun.model_validate(snapshot.to_dict())
        if stored.model_dump() != run.model_dump():
            raise EvaluationRunError("run_id is already bound to another configuration")
        return stored

class InMemoryEvaluationJobDispatcher:
    """Dispatch seam for Cloud Run Jobs; records each unique job submission."""

    def __init__(self) -> None:
        self.dispatched: dict[str, tuple[EvaluationRun, EvaluationDataset]] = {}

    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None:
        self.dispatched.setdefault(run.run_id, (run.model_copy(deep=True), dataset.model_copy(deep=True)))


@dataclass(frozen=True)
class EvaluationOutcome:
    """Internal, content-free outcome produced by one evaluator invocation."""

    retrieval_success: bool
    grounded: bool
    citation_correct: bool
    answer_quality: bool
    refusal_correct: bool
    response_time_ms: float


class EvaluationWorker:
    """Execute one dispatched run without persisting interaction content."""

    def __init__(self, runs: EvaluationRunStore, *, response_time_threshold_ms: float = 15_000) -> None:
        self.runs = runs
        self.response_time_threshold_ms = response_time_threshold_ms

    def execute(self, run_id: str, dataset: EvaluationDataset, evaluate: Callable[[str], EvaluationOutcome]) -> EvaluationRun:
        run = self.runs.get(run_id)
        if run is None:
            raise EvaluationRunError("unknown evaluation run")
        if run.status == "completed":
            return run
        running = run.model_copy(update={"status": "running"})
        self.runs.update(running)
        try:
            item_ids = set(run.selected_item_ids)
            if item_ids - {item.item_id for item in dataset.items}:
                raise EvaluationRunError("run contains items outside its dataset version")
            results = [evaluate(item_id) for item_id in run.selected_item_ids]
            metrics = [
                _metric("retrieval_success", results, lambda result: result.retrieval_success, threshold=90),
                _metric("grounded_claims", results, lambda result: result.grounded, threshold=95),
                _metric("citation_correctness", results, lambda result: result.citation_correct, threshold=95),
                _metric("answer_quality", results, lambda result: result.answer_quality, threshold=85),
                _metric("refusal_correctness", results, lambda result: result.refusal_correct, threshold=95),
                _metric(
                    "response_time",
                    results,
                    lambda result: result.response_time_ms <= self.response_time_threshold_ms,
                    threshold=95,
                ),
            ]
            item_results = [
                EvaluationItemResult(item_id=item_id, **result.__dict__)
                for item_id, result in zip(run.selected_item_ids, results, strict=True)
            ]
            return self.runs.update(
                running.model_copy(update={"status": "completed", "metrics": metrics, "item_results": item_results})
            )
        except Exception as exc:
            return self.runs.update(running.model_copy(update={"status": "failed", "error": str(exc)[:500]}))


def select_items(dataset: EvaluationDataset, item_ids: list[str] | None) -> tuple[str, ...]:
    available = {item.item_id for item in dataset.items}
    if item_ids is None:
        return tuple(item.item_id for item in dataset.items)
    selected = tuple(dict.fromkeys(item_ids))
    if not selected:
        raise EvaluationRunError("item_ids must not be empty for a subset run")
    unknown = set(selected) - available
    if unknown:
        raise EvaluationRunError(f"unknown evaluation item IDs: {', '.join(sorted(unknown))}")
    return selected


def stable_run_id(
    dataset_version_id: str,
    corpus_version_id: str,
    configuration_version_id: str,
    model_version_id: str,
    selected_item_ids: tuple[str, ...],
) -> str:
    material = "|".join((dataset_version_id, corpus_version_id, configuration_version_id, model_version_id, *selected_item_ids))
    return "eval-" + sha256(material.encode("utf-8")).hexdigest()[:32]


class EvaluationJobService:
    def __init__(self, datasets: ActiveEvaluationDatasetStore, runs: EvaluationRunStore, dispatcher: EvaluationJobDispatcher) -> None:
        self.datasets = datasets
        self.runs = runs
        self.dispatcher = dispatcher

    def launch(
        self,
        *,
        corpus_version_id: str,
        configuration_version_id: str,
        model_version_id: str,
        item_ids: list[str] | None,
    ) -> EvaluationRun:
        dataset = self.datasets.active_dataset()
        if dataset is None:
            raise EvaluationRunError("no active evaluation dataset")
        dataset_version_id = dataset.dataset_version_id
        selected = select_items(dataset, item_ids)
        run = EvaluationRun(
            run_id=stable_run_id(dataset_version_id, corpus_version_id, configuration_version_id, model_version_id, selected),
            dataset_version_id=dataset_version_id,
            corpus_version_id=corpus_version_id,
            status="queued",
            selected_item_ids=list(selected),
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
            report_eligibility=self.datasets.report_eligibility(dataset_version_id),
        )
        stored = self.runs.create_or_get(run)
        if stored.status == "queued":
            self.dispatcher.dispatch(stored, dataset)
        return stored

    def status(self, run_id: str) -> EvaluationRun:
        run = self.runs.get(run_id)
        if run is None:
            raise EvaluationRunError("unknown evaluation run")
        return run.model_copy(update={"report_eligibility": self.datasets.report_eligibility(run.dataset_version_id)})


def _metric(
    name: str,
    results: list[EvaluationOutcome],
    predicate: Callable[[EvaluationOutcome], bool],
    *,
    threshold: float,
) -> EvaluationMetric:
    numerator = sum(predicate(result) for result in results)
    denominator = len(results)
    percentage = round(100 * numerator / denominator, 2) if denominator else 0.0
    return EvaluationMetric(
        metric_name=name,
        numerator=numerator,
        denominator=denominator,
        percentage=percentage,
        threshold=threshold,
        passed=percentage >= threshold,
    )


def main() -> None:
    """Fail clearly until the production evaluation worker is configured."""
    raise RuntimeError(
        "The evaluation Cloud Run Job is not configured with a durable dataset, "
        "run store, and evaluator."
    )


if __name__ == "__main__":
    main()
