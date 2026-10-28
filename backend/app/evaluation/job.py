"""Evaluation run identity, selection, dispatch, and idempotent status storage."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from threading import RLock
from typing import Any, Literal, Protocol
import dataclasses
from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime, timezone

from google.api_core.exceptions import AlreadyExists

from app.contracts import (
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItemResult,
    EvaluationMetric,
    EvaluationReportEligibility,
    EvaluationRun,
    RetrievalResult,
    RunBoundSupportEvidence,
)
from app.contracts.api import AnswerOutcomeKind
from app.evaluation.dataset import EvaluationDatasetStore
from app.evaluation.support import approved_support_chunk_ids


class EvaluationRunError(ValueError):
    """Raised when a run request cannot be bound to immutable inputs."""


class EvaluationInterrupted(RuntimeError):
    """Raised when a worker receives termination before a run completes."""


class EvaluationStageError(EvaluationRunError):
    """A typed stage failure that can retain retrieval evidence."""

    def __init__(self, message: str, *, retrieval: RetrievalResult | None = None) -> None:
        super().__init__(message)
        self.retrieval = retrieval


class EvaluationReviewError(EvaluationRunError):
    """Raised when an assessment cannot bind to the exact generated output."""


class EvaluationRunStore(Protocol):
    def create_or_get(self, run: EvaluationRun) -> EvaluationRun: ...
    def get(self, run_id: str) -> EvaluationRun | None: ...
    def update(self, run: EvaluationRun) -> EvaluationRun: ...


@dataclass(frozen=True)
class AggregateRunStatus:
    """Content-free status projection for the simple evaluation workflow."""

    run_id: str
    dataset_version_id: str
    status: Literal["queued", "running", "completed", "failed"]
    metrics: tuple[EvaluationMetric, ...] = ()
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class AggregateRunStatusStore(Protocol):
    def status(self, run_id: str) -> AggregateRunStatus | None: ...

    def update_status(self, status: AggregateRunStatus) -> AggregateRunStatus: ...


AssessmentStatus = Literal["awaiting_review", "reviewed", "inapplicable"]


@dataclass(frozen=True)
class EvaluationItemStage:
    """Durable, content-free state for the independent stages of one item."""

    run_id: str
    item_id: str
    retrieval: RetrievalResult | None = None
    retrieval_status: Literal["pending", "running", "succeeded", "failed", "unavailable"] = "pending"
    generation_status: Literal["pending", "running", "succeeded", "failed", "unavailable"] = "pending"
    generation_outcome: AnswerOutcomeKind | None = None
    assessment_status: AssessmentStatus = "awaiting_review"
    output_handle: str | None = None
    output_digest: str | None = None
    operational_error: str | None = None
    retrieval_success: bool = False
    grounded: bool = False
    citation_correct: bool = False
    answer_quality: bool = False
    refusal_correct: bool = False
    response_time_ms: float = 0.0

    @property
    def ready_for_finalization(self) -> bool:
        return (
            self.assessment_status in {"reviewed", "inapplicable"}
            and self.retrieval_status != "failed"
            and self.generation_status != "failed"
            and self.operational_error is None
        )


class EvaluationStageStore(Protocol):
    def get(self, run_id: str, item_id: str) -> EvaluationItemStage | None: ...
    def update(self, stage: EvaluationItemStage) -> EvaluationItemStage: ...


class InMemoryEvaluationStageStore:
    """Restart-safe seam used by tests and local workers."""

    def __init__(self) -> None:
        self._stages: dict[tuple[str, str], EvaluationItemStage] = {}
        self._lock = RLock()

    def get(self, run_id: str, item_id: str) -> EvaluationItemStage | None:
        with self._lock:
            return deepcopy(self._stages.get((run_id, item_id)))

    def update(self, stage: EvaluationItemStage) -> EvaluationItemStage:
        with self._lock:
            self._stages[(stage.run_id, stage.item_id)] = stage
            return deepcopy(stage)


class FirestoreEvaluationStageStore:
    """Persist item stages separately from the public run summary."""

    collection_id = "evaluation_run_stages"

    def __init__(self, client: Any) -> None:
        self.stages = client.collection(self.collection_id)

    def get(self, run_id: str, item_id: str) -> EvaluationItemStage | None:
        snapshot = self.stages.document(_stage_id(run_id, item_id)).get()
        if not snapshot.exists:
            return None
        return _stage_from_payload(snapshot.to_dict())

    def update(self, stage: EvaluationItemStage) -> EvaluationItemStage:
        self.stages.document(_stage_id(stage.run_id, stage.item_id)).set(_stage_payload(stage))
        return stage


class ActiveEvaluationDatasetStore(EvaluationDatasetStore, Protocol):
    """Dataset storage that can resolve the sole generation visible to launches."""

    def active_dataset(self) -> EvaluationDataset | None: ...

    def report_eligibility(self, generation_id: str) -> EvaluationReportEligibility: ...


class RunSupportEvidenceStore(Protocol):
    def get_evidence(self, *, run_id: str, item_id: str) -> RunBoundSupportEvidence | None: ...

    def get_marker(self, *, run_id: str) -> object | None: ...


class EvaluationJobDispatcher(Protocol):
    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None: ...


class EvaluationCandidateSnapshotResolver(Protocol):
    """Resolves the complete candidate identity before a run is persisted."""

    def resolve(
        self,
        *,
        corpus_version_id: str,
        configuration_version_id: str,
        model_version_id: str,
    ) -> EvaluationCandidateSnapshot: ...

    def resolve_candidate(self) -> "ResolvedEvaluationCandidate": ...


@dataclass(frozen=True)
class ResolvedEvaluationCandidate:
    """The server-owned identity used to bind a newly launched run."""

    corpus_version_id: str
    configuration_version_id: str
    model_version_id: str
    snapshot: EvaluationCandidateSnapshot


class InMemoryEvaluationRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, EvaluationRun] = {}
        self._statuses: dict[str, AggregateRunStatus] = {}
        self._lock = RLock()

    def create_or_get(self, run: EvaluationRun) -> EvaluationRun:
        with self._lock:
            existing = self._runs.get(run.run_id)
            if existing is not None:
                if existing.model_dump() != run.model_dump():
                    raise EvaluationRunError("run_id is already bound to another configuration")
                return deepcopy(existing)
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._statuses.setdefault(
                run.run_id,
                AggregateRunStatus(run_id=run.run_id, dataset_version_id=run.dataset_version_id, status=run.status),
            )
            return deepcopy(run)

    def get(self, run_id: str) -> EvaluationRun | None:
        with self._lock:
            return deepcopy(self._runs.get(run_id))

    def update(self, run: EvaluationRun) -> EvaluationRun:
        with self._lock:
            if run.run_id not in self._runs:
                raise EvaluationRunError("unknown evaluation run")
            existing = self._statuses.get(run.run_id)
            if existing is not None and (
                existing.status == "failed" or (existing.status == "completed" and existing.metrics)
            ):
                return deepcopy(self._runs[run.run_id])
            self._runs[run.run_id] = run.model_copy(deep=True)
            self._statuses[run.run_id] = AggregateRunStatus(
                run_id=run.run_id,
                dataset_version_id=run.dataset_version_id,
                status=run.status,
                metrics=tuple(run.metrics),
                started_at=(existing.started_at if existing and existing.started_at else datetime.now(timezone.utc) if run.status == "running" else None),
                completed_at=(datetime.now(timezone.utc) if run.status in {"completed", "failed"} else None),
                error=run.error,
            )
            return deepcopy(run)

    def status(self, run_id: str) -> AggregateRunStatus | None:
        with self._lock:
            return deepcopy(self._statuses.get(run_id))

    def update_status(self, status: AggregateRunStatus) -> AggregateRunStatus:
        with self._lock:
            current = self._statuses.get(status.run_id)
            if current is not None and current.status in {"completed", "failed"}:
                return deepcopy(current)
            self._statuses[status.run_id] = status
            return deepcopy(status)


class FirestoreEvaluationRunStore:
    """Durable immutable run records, separated from evaluator-authored data."""

    collection_id = "evaluation_runs"
    status_collection_id = "evaluation_run_status"

    def __init__(self, client: Any) -> None:
        self.runs = client.collection(self.collection_id)
        self.statuses = client.collection(self.status_collection_id)

    def create_or_get(self, run: EvaluationRun) -> EvaluationRun:
        reference = self.runs.document(run.run_id)
        snapshot = reference.get()
        if snapshot.exists:
            return self._same_or_conflict(snapshot, run)
        try:
            reference.create(run.model_dump(mode="json"))
            self.update_status(AggregateRunStatus(run_id=run.run_id, dataset_version_id=run.dataset_version_id, status=run.status))
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
        existing = self.status(run.run_id)
        if existing is not None and (
            existing.status == "failed" or (existing.status == "completed" and existing.metrics)
        ):
            stored = reference.get()
            return EvaluationRun.model_validate(stored.to_dict())
        reference.set(run.model_dump(mode="json"))
        self.update_status(AggregateRunStatus(
            run_id=run.run_id,
            dataset_version_id=run.dataset_version_id,
            status=run.status,
            metrics=tuple(run.metrics),
            started_at=(existing.started_at if existing and existing.started_at else datetime.now(timezone.utc) if run.status == "running" else None),
            completed_at=(datetime.now(timezone.utc) if run.status in {"completed", "failed"} else None),
            error=run.error,
        ))
        return run.model_copy(deep=True)

    def status(self, run_id: str) -> AggregateRunStatus | None:
        snapshot = self.statuses.document(run_id).get()
        if not snapshot.exists:
            return None
        payload = snapshot.to_dict()
        return AggregateRunStatus(
            run_id=run_id,
            dataset_version_id=str(payload["dataset_version_id"]),
            status=payload["status"],
            metrics=tuple(EvaluationMetric.model_validate(item) for item in payload.get("metrics", [])),
            started_at=_datetime(payload.get("started_at")),
            completed_at=_datetime(payload.get("completed_at")),
            error=payload.get("error"),
        )

    def update_status(self, status: AggregateRunStatus) -> AggregateRunStatus:
        current = self.status(status.run_id)
        if current is not None and (
            current.status == "failed" or (current.status == "completed" and current.metrics)
        ):
            return current
        payload = {
            "run_id": status.run_id,
            "dataset_version_id": status.dataset_version_id,
            "status": status.status,
            "metrics": [metric.model_dump(mode="json") for metric in status.metrics],
            "started_at": status.started_at.isoformat() if status.started_at else None,
            "completed_at": status.completed_at.isoformat() if status.completed_at else None,
            "error": status.error,
        }
        self.statuses.document(status.run_id).set(payload)
        return status

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
    applicable_metrics: frozenset[str] = frozenset({
        "retrieval_success", "grounded_claims", "citation_correctness", "answer_quality", "refusal_correctness",
    })
    retrieval: RetrievalResult | None = None
    generation_status: Literal["succeeded", "failed", "unavailable"] = "succeeded"
    generation_outcome: AnswerOutcomeKind | None = "answer"
    output_handle: str | None = None
    output_digest: str | None = None
    # Existing pure outcomes are already assessed. Real generated outputs set
    # this to awaiting_review until an exact-digest review is submitted.
    assessment_status: AssessmentStatus = "reviewed"
    operational_error: str | None = None


class EvaluationWorker:
    """Execute one dispatched run without persisting interaction content."""

    def __init__(
        self,
        runs: EvaluationRunStore,
        stages: EvaluationStageStore | None = None,
        *,
        response_time_threshold_ms: float = 15_000,
    ) -> None:
        self.runs = runs
        self.stages = stages or InMemoryEvaluationStageStore()
        self.response_time_threshold_ms = response_time_threshold_ms

    def execute(
        self,
        run_id: str,
        dataset: EvaluationDataset,
        evaluate: Callable[[str], EvaluationOutcome],
        *,
        termination_requested: Callable[[], bool] | None = None,
        support_preparer: Callable[[EvaluationRun, EvaluationDataset], object] | None = None,
        support_evidence: Callable[..., RunBoundSupportEvidence | None] | None = None,
    ) -> EvaluationRun:
        run = self.runs.get(run_id)
        if run is None:
            raise EvaluationRunError("unknown evaluation run")
        # Terminal states are immutable.  Cloud Run may replay a successful
        # invocation after the acknowledgement was lost; replaying must not
        # re-score items or overwrite the terminal summary.
        if run.status in {"completed", "failed"}:
            return run
        running = run.model_copy(update={"status": "running"})
        self.runs.update(running)
        try:
            if dataset.dataset_version_id != run.dataset_version_id:
                raise EvaluationRunError("worker dataset does not match the run-pinned generation")
            # A simple evaluation always evaluates the accepted dataset in
            # source order.  Candidate-pinned runs retain their historical
            # subset semantics for the deferred release-evidence workflow.
            item_ids_in_dataset = [item.item_id for item in dataset.items]
            selected_item_ids = (
                item_ids_in_dataset
                if run.candidate_snapshot is None
                else run.selected_item_ids
            )
            if set(selected_item_ids) - set(item_ids_in_dataset):
                raise EvaluationRunError("run contains items outside its dataset version")
            if support_preparer is not None:
                support_preparer(run, dataset)
            results = []
            for item_id in selected_item_ids:
                if support_evidence is not None and support_evidence(run_id=run.run_id, item_id=item_id) is None:
                    raise EvaluationRunError("run-bound support evidence is incomplete")
                if termination_requested is not None and termination_requested():
                    raise EvaluationInterrupted("evaluation interrupted before next item")
                try:
                    outcome = evaluate(item_id)
                except Exception as exc:
                    previous = self.stages.get(run_id, item_id)
                    base = previous or EvaluationItemStage(run_id=run_id, item_id=item_id)
                    retrieval = getattr(exc, "retrieval", None)
                    self.stages.update(dataclasses.replace(
                        base,
                        retrieval=retrieval or base.retrieval,
                        retrieval_status="succeeded" if (retrieval or base.retrieval) is not None else base.retrieval_status,
                        generation_status="failed",
                        assessment_status="awaiting_review",
                        operational_error=type(exc).__name__,
                    ))
                    raise
                self.stages.update(_stage_from_outcome(run_id, item_id, outcome))
                results.append(outcome)
            if termination_requested is not None and termination_requested():
                raise EvaluationInterrupted("evaluation interrupted after item")
            metrics = (
                []
                if any(result.assessment_status == "awaiting_review" for result in results)
                else _metrics(results, self.response_time_threshold_ms)
            )
            item_results = [
                EvaluationItemResult(
                    item_id=item_id,
                    retrieval_success=result.retrieval_success,
                    grounded=result.grounded,
                    citation_correct=result.citation_correct,
                    answer_quality=result.answer_quality,
                    refusal_correct=result.refusal_correct,
                    response_time_ms=result.response_time_ms,
                )
                for item_id, result in zip(selected_item_ids, results, strict=True)
            ]
            return self.runs.update(
                running.model_copy(update={"status": "completed", "metrics": metrics, "item_results": item_results})
            )
        except EvaluationInterrupted as exc:
            self.runs.update(
                running.model_copy(
                    update={"status": "failed", "error": f"{type(exc).__name__}: evaluation interrupted"}
                )
            )
            raise
        except Exception as exc:
            # Provider and evaluator errors can include question or answer text;
            # keep durable run state content-free.
            return self.runs.update(
                running.model_copy(update={"status": "failed", "error": f"{type(exc).__name__}: evaluation failed"})
            )

    def finalize(
        self,
        run_id: str,
        *,
        cleanup_output: Callable[[str], None] | None = None,
    ) -> EvaluationRun:
        """Recompute all six gates after exact-output review is complete.

        The worker never infers a review from a raw answer. Every selected
        item must have a durable stage and a matching reviewed assessment;
        otherwise the run remains ineligible and no raw output is cleaned up.
        """

        run = self.runs.get(run_id)
        if run is None:
            raise EvaluationRunError("unknown evaluation run")
        stages = [self.stages.get(run_id, item_id) for item_id in run.selected_item_ids]
        if any(stage is None for stage in stages):
            return self._ineligible(run, "evaluation assessment is incomplete")
        resolved = [stage for stage in stages if stage is not None]
        if any(not stage.ready_for_finalization for stage in resolved):
            return self._ineligible(run, "evaluation assessment is incomplete or operationally failed")

        outcomes = [
            EvaluationOutcome(
                retrieval_success=stage.retrieval_success,
                grounded=stage.grounded,
                citation_correct=stage.citation_correct,
                answer_quality=stage.answer_quality,
                refusal_correct=stage.refusal_correct,
                response_time_ms=stage.response_time_ms,
                retrieval=stage.retrieval,
                generation_status=stage.generation_status,
                generation_outcome=stage.generation_outcome,
                output_handle=stage.output_handle,
                output_digest=stage.output_digest,
                assessment_status=stage.assessment_status,
                operational_error=stage.operational_error,
            )
            for stage in resolved
        ]
        metrics = _metrics(outcomes, self.response_time_threshold_ms)
        if cleanup_output is not None:
            for stage in resolved:
                if stage.output_handle is not None:
                    cleanup_output(stage.output_handle)
        return self.runs.update(run.model_copy(update={"status": "completed", "metrics": metrics}))

    def review_item(
        self,
        run_id: str,
        item_id: str,
        *,
        output_digest: str,
        assessment: EvaluationOutcome,
    ) -> EvaluationItemStage:
        """Bind one human assessment to the exact evaluator output digest."""

        stage = self.stages.get(run_id, item_id)
        if stage is None:
            raise EvaluationReviewError("evaluation item stage is unavailable")
        if stage.output_handle is None or stage.output_digest is None:
            raise EvaluationReviewError("evaluation item has no reviewable output handle")
        if output_digest != stage.output_digest:
            raise EvaluationReviewError("assessment digest does not match the generated output")
        reviewed = dataclasses.replace(
            stage,
            assessment_status="reviewed",
            grounded=assessment.grounded,
            citation_correct=assessment.citation_correct,
            answer_quality=assessment.answer_quality,
            refusal_correct=assessment.refusal_correct,
        )
        return self.stages.update(reviewed)

    def _ineligible(self, run: EvaluationRun, error: str) -> EvaluationRun:
        # Keep the run durable and content-free. Empty/old metrics cannot be
        # mistaken for a finalized report by the existing release checks.
        return self.runs.update(run.model_copy(update={"metrics": [], "error": error}))


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
    configuration_version_id: str,
    model_version_id: str,
    candidate_snapshot: EvaluationCandidateSnapshot,
    selected_item_ids: tuple[str, ...],
) -> str:
    material = "|".join((
        dataset_version_id,
        configuration_version_id,
        model_version_id,
        candidate_snapshot.model_dump_json(),
        *selected_item_ids,
    ))
    return "eval-" + sha256(material.encode("utf-8")).hexdigest()[:32]


class EvaluationJobService:
    def __init__(
        self,
        datasets: ActiveEvaluationDatasetStore,
        runs: EvaluationRunStore,
        dispatcher: EvaluationJobDispatcher,
        snapshots: EvaluationCandidateSnapshotResolver,
        support_evidence: RunSupportEvidenceStore | None = None,
    ) -> None:
        self.datasets = datasets
        self.runs = runs
        self.dispatcher = dispatcher
        self.snapshots = snapshots
        self.support_evidence = support_evidence

    def launch(
        self,
        *,
        corpus_version_id: str | None = None,
        configuration_version_id: str | None = None,
        model_version_id: str | None = None,
        item_ids: list[str] | None,
    ) -> EvaluationRun:
        dataset = self.datasets.active_dataset()
        if dataset is None:
            raise EvaluationRunError("no active evaluation dataset")
        dataset_version_id = dataset.dataset_version_id
        selected = select_items(dataset, item_ids)
        if all(value is None for value in (corpus_version_id, configuration_version_id, model_version_id)):
            candidate = self.snapshots.resolve_candidate()
            corpus_version_id = candidate.corpus_version_id
            configuration_version_id = candidate.configuration_version_id
            model_version_id = candidate.model_version_id
            snapshot = candidate.snapshot
        elif all(value is not None for value in (corpus_version_id, configuration_version_id, model_version_id)):
            snapshot = self.snapshots.resolve(
                corpus_version_id=corpus_version_id,
                configuration_version_id=configuration_version_id,
                model_version_id=model_version_id,
            )
        else:
            raise EvaluationRunError("candidate identity must be fully specified or server-resolved")
        if snapshot.corpus_binding.corpus_version_id != corpus_version_id:
            raise EvaluationRunError("candidate snapshot does not match the requested corpus version")
        run = EvaluationRun(
            run_id=stable_run_id(dataset_version_id, configuration_version_id, model_version_id, snapshot, selected),
            dataset_version_id=dataset_version_id,
            corpus_version_id=corpus_version_id,
            status="queued",
            selected_item_ids=list(selected),
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
            candidate_snapshot=snapshot,
            report_eligibility=None,
        )
        run = run.model_copy(update={"report_eligibility": self._eligibility_for_run(run)})
        stored = self.runs.create_or_get(run)
        if stored.status == "queued":
            self.dispatcher.dispatch(stored, dataset)
        return stored

    def status(self, run_id: str) -> EvaluationRun:
        run = self.runs.get(run_id)
        if run is None:
            raise EvaluationRunError("unknown evaluation run")
        return run.model_copy(update={
            "report_eligibility": self._eligibility_for_run(run),
        })

    def _eligibility_for_run(self, run: EvaluationRun) -> EvaluationReportEligibility:
        dataset = self.datasets.get(run.dataset_version_id)
        if dataset is None:
            raise EvaluationRunError("evaluation dataset is unavailable")
        eligibility = self._eligibility_for_selection(
            run.dataset_version_id, tuple(run.selected_item_ids), len(dataset.items)
        )
        if self.support_evidence is None:
            return eligibility
        if not eligibility.is_current_generation or len(run.selected_item_ids) != len(dataset.items):
            return eligibility.model_copy(update={"eligibility": "exploratory", "approved_gold_count": 0})
        marker = self.support_evidence.get_marker(run_id=run.run_id)
        if marker is None:
            return eligibility.model_copy(update={"eligibility": "exploratory", "approved_gold_count": 0})
        approved = 0
        for item_id in run.selected_item_ids:
            evidence = self.support_evidence.get_evidence(run_id=run.run_id, item_id=item_id)
            if approved_support_chunk_ids(evidence, run=run):
                approved += 1
        return eligibility.model_copy(update={
            "approved_gold_count": approved,
            "eligibility": "release_evidence" if 30 <= approved <= 50 else "exploratory",
        })

    def _eligibility_for_selection(
        self, dataset_version_id: str, selected_item_ids: tuple[str, ...], item_count: int
    ) -> EvaluationReportEligibility:
        eligibility = self.datasets.report_eligibility(dataset_version_id)
        if len(selected_item_ids) == item_count:
            return eligibility
        return eligibility.model_copy(update={
            "eligibility": "exploratory",
        })


def _metric(
    name: str,
    results: list[EvaluationOutcome],
    predicate: Callable[[EvaluationOutcome], bool],
    *,
    threshold: float,
) -> EvaluationMetric:
    applicable = [
        result for result in results
        if name == "response_time" or name in result.applicable_metrics
    ]
    numerator = sum(predicate(result) for result in applicable)
    denominator = len(applicable)
    percentage = round(100 * numerator / denominator, 2) if denominator else 0.0
    return EvaluationMetric(
        metric_name=name,
        numerator=numerator,
        denominator=denominator,
        percentage=percentage,
        threshold=threshold,
        passed=denominator > 0 and percentage >= threshold,
    )


def _metrics(results: list[EvaluationOutcome], response_time_threshold_ms: float) -> list[EvaluationMetric]:
    return [
        _metric("retrieval_success", results, lambda result: result.retrieval_success, threshold=90),
        _metric("grounded_claims", results, lambda result: result.grounded, threshold=95),
        _metric("citation_correctness", results, lambda result: result.citation_correct, threshold=95),
        _metric("answer_quality", results, lambda result: result.answer_quality, threshold=85),
        _metric("refusal_correctness", results, lambda result: result.refusal_correct, threshold=95),
        _metric(
            "response_time", results,
            lambda result: result.response_time_ms <= response_time_threshold_ms,
            threshold=95,
        ),
    ]


def _stage_id(run_id: str, item_id: str) -> str:
    return f"{run_id}:{item_id}"


def _datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _stage_from_outcome(run_id: str, item_id: str, outcome: EvaluationOutcome) -> EvaluationItemStage:
    retrieval_status = "succeeded" if outcome.retrieval is not None else "unavailable"
    return EvaluationItemStage(
        run_id=run_id,
        item_id=item_id,
        retrieval=outcome.retrieval,
        retrieval_status=retrieval_status,
        generation_status=outcome.generation_status,
        generation_outcome=outcome.generation_outcome,
        assessment_status=outcome.assessment_status,
        output_handle=outcome.output_handle,
        output_digest=outcome.output_digest,
        operational_error=outcome.operational_error,
        retrieval_success=outcome.retrieval_success,
        grounded=outcome.grounded,
        citation_correct=outcome.citation_correct,
        answer_quality=outcome.answer_quality,
        refusal_correct=outcome.refusal_correct,
        response_time_ms=outcome.response_time_ms,
    )


def _stage_payload(stage: EvaluationItemStage) -> dict[str, Any]:
    payload = stage.__dict__.copy()
    payload["retrieval"] = stage.retrieval.model_dump(mode="json") if stage.retrieval is not None else None
    return payload


def _stage_from_payload(payload: dict[str, Any]) -> EvaluationItemStage:
    values = dict(payload)
    if values.get("retrieval") is not None:
        values["retrieval"] = RetrievalResult.model_validate(values["retrieval"])
    return EvaluationItemStage(**values)


def main() -> None:
    """Fail clearly until the production evaluation worker is configured."""
    raise RuntimeError(
        "The evaluation Cloud Run Job is not configured with a durable dataset, "
        "run store, and evaluator."
    )


if __name__ == "__main__":
    main()
