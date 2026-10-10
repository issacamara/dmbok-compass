"""Immutable sponsor release decisions bound to evaluation evidence."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Literal, Protocol

from pydantic import Field

from app.contracts import EvaluationReportEligibility, EvaluationRun, ReleaseDecision
from app.evaluation import EvaluationRunStore
from app.contracts.api import ContractModel


class ReleaseDecisionError(ValueError):
    """Raised when a release decision cannot be safely recorded."""


class ReleaseDecisionRequest(ContractModel):
    release_id: str = Field(min_length=1, max_length=128)
    evaluation_run_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    corpus_version_id: str = Field(min_length=1, max_length=128)
    configuration_version_id: str = Field(min_length=1, max_length=128)
    provider_version_id: str = Field(min_length=1, max_length=128)
    model_version_id: str = Field(min_length=1, max_length=128)
    scorer_version_id: str = Field(min_length=1, max_length=128)
    gate_report_ids: list[str] = Field(min_length=1, max_length=100)
    decision: Literal["approved", "rejected"]
    rationale: str = Field(min_length=1, max_length=5000)
    exception_approved: bool = False
    exception_rationale: str | None = Field(default=None, max_length=5000)

    def to_decision(self) -> ReleaseDecision:
        return ReleaseDecision(**self.model_dump())


class ReleaseDecisionStore(Protocol):
    def create_or_get(self, decision: ReleaseDecision) -> ReleaseDecision: ...

    def get(self, release_id: str) -> ReleaseDecision | None: ...


class ReleaseEvidenceStore(Protocol):
    """Resolve current, transactionally consistent eligibility for a run."""

    def report_eligibility(self, generation_id: str) -> EvaluationReportEligibility: ...


class InMemoryReleaseDecisionStore:
    """Reference store with create-once semantics for immutable decisions."""

    def __init__(self) -> None:
        self._decisions: dict[str, ReleaseDecision] = {}
        self._lock = RLock()

    def create_or_get(self, decision: ReleaseDecision) -> ReleaseDecision:
        with self._lock:
            existing = self._decisions.get(decision.release_id)
            if existing is not None:
                if existing != decision:
                    raise ReleaseDecisionError("release_id is already bound to another decision")
                return deepcopy(existing)
            self._decisions[decision.release_id] = decision.model_copy(deep=True)
            return deepcopy(decision)

    def get(self, release_id: str) -> ReleaseDecision | None:
        with self._lock:
            decision = self._decisions.get(release_id)
            return deepcopy(decision) if decision is not None else None


class ReleaseDecisionService:
    """Validate and persist a decision against one immutable evaluation run."""

    _REQUIRED_METRICS = frozenset({
        "retrieval_success",
        "grounded_claims",
        "citation_correctness",
        "answer_quality",
        "refusal_correctness",
        "response_time",
    })

    def __init__(
        self,
        runs: EvaluationRunStore,
        decisions: ReleaseDecisionStore,
        evidence: ReleaseEvidenceStore,
    ) -> None:
        self.runs = runs
        self.decisions = decisions
        self.evidence = evidence

    def record(self, request: ReleaseDecisionRequest) -> ReleaseDecision:
        run = self.runs.get(request.evaluation_run_id)
        if run is None:
            raise ReleaseDecisionError("evaluation evidence is missing")
        self._validate_binding(request, run)
        if request.decision == "approved":
            self._validate_current_evidence(run)
            self._validate_gates(request, run)
        return self.decisions.create_or_get(request.to_decision())

    def get(self, release_id: str) -> ReleaseDecision:
        decision = self.decisions.get(release_id)
        if decision is None:
            raise ReleaseDecisionError("release decision was not found")
        return decision

    @staticmethod
    def _validate_binding(request: ReleaseDecisionRequest, run: EvaluationRun) -> None:
        expected = {
            "dataset_version_id": run.dataset_version_id,
            "corpus_version_id": run.corpus_version_id,
            "configuration_version_id": run.configuration_version_id,
            "model_version_id": run.model_version_id,
        }
        actual = request.model_dump(include=set(expected))
        if actual != expected:
            raise ReleaseDecisionError("release evidence is stale or does not match the evaluation run")
        if run.status != "completed":
            raise ReleaseDecisionError("release evidence is missing or the evaluation run is not complete")
        if not request.gate_report_ids or len(request.gate_report_ids) != len(set(request.gate_report_ids)):
            raise ReleaseDecisionError("release gate report evidence is missing or duplicated")

    def _validate_current_evidence(self, run: EvaluationRun) -> None:
        eligibility = self.evidence.report_eligibility(run.dataset_version_id)
        if (
            eligibility.generation_id != run.dataset_version_id
            or eligibility.eligibility != "release_evidence"
            or not eligibility.is_current_generation
            or eligibility.is_superseded
            or not 30 <= eligibility.approved_gold_count <= 50
        ):
            raise ReleaseDecisionError("release evidence is exploratory, superseded, or lacks 30–50 approved gold items")

    def _validate_gates(self, request: ReleaseDecisionRequest, run: EvaluationRun) -> None:
        metrics = {metric.metric_name: metric for metric in run.metrics}
        if len(run.metrics) != len(self._REQUIRED_METRICS) or set(metrics) != self._REQUIRED_METRICS:
            raise ReleaseDecisionError("release evidence must include all six required metrics")
        if not all(metric.passed for metric in metrics.values()) and not request.exception_approved:
            raise ReleaseDecisionError("one or more release gates failed")
