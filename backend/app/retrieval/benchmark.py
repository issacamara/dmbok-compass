"""Offline, deterministic evaluation for a pinned dense retrieval candidate."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from app.contracts import CandidateIdentity, EvaluationCase, RetrievalResult
from app.retrieval.evidence import (
    MAX_EVIDENCE_PASSAGES,
    PARTIAL_EVIDENCE_THRESHOLD,
    STRONG_EVIDENCE_THRESHOLD,
)


class RetrievalRunner(Protocol):
    def retrieve(self, case: EvaluationCase) -> RetrievalResult: ...


@dataclass(frozen=True)
class RetrievalMetrics:
    count: int
    hit_at_5: float
    recall_at_5: float
    mrr: float
    ndcg: float
    zero_hits: int
    slices: Mapping[str, Mapping[str, float | int]]


@dataclass(frozen=True)
class CalibrationReport:
    tuning_digest: str
    held_out_digest: str
    selected_threshold: float
    policy_version: str
    false_refusals: int
    unsupported_answer_inputs: int


@dataclass(frozen=True)
class BenchmarkReport:
    candidate: CandidateIdentity
    dataset_id: str
    strategy: str
    tuning_digest: str
    held_out_digest: str
    metrics: RetrievalMetrics
    calibration: CalibrationReport

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.model_dump(mode="json"),
            "dataset_id": self.dataset_id,
            "strategy": self.strategy,
            "tuning_digest": self.tuning_digest,
            "held_out_digest": self.held_out_digest,
            "metrics": _as_json(self.metrics),
            "calibration": _as_json(self.calibration),
            "gate": {"name": "hit@5", "required_percentage": 90.0},
        }


def evaluate(
    cases: Iterable[EvaluationCase],
    retriever: RetrievalRunner | Callable[[EvaluationCase], RetrievalResult],
    *,
    candidate: CandidateIdentity,
    dataset_id: str,
    tuning_cases: Iterable[EvaluationCase] = (),
    held_out_cases: Iterable[EvaluationCase] = (),
    strategy: str = "dense-v1",
) -> BenchmarkReport:
    """Evaluate cases and calibrate only from the explicitly supplied tuning split."""

    cases = _freeze_cases(cases)
    tuning = _freeze_cases(tuning_cases)
    held_out = _freeze_cases(held_out_cases)
    _validate_splits(tuning, held_out, cases)
    results = tuple(_retrieve(retriever, case) for case in cases)
    tuning_results = tuple(_retrieve(retriever, case) for case in tuning)
    held_out_results = tuple(_retrieve(retriever, case) for case in held_out)
    _validate_candidate(results + tuning_results + held_out_results, candidate)

    threshold = calibrate_threshold(tuning, tuning_results)
    calibration = CalibrationReport(
        tuning_digest=split_digest(tuning),
        held_out_digest=split_digest(held_out),
        selected_threshold=threshold,
        policy_version=f"dense-evidence-{threshold:.2f}",
        false_refusals=_false_refusals(held_out, held_out_results, threshold),
        unsupported_answer_inputs=_unsupported_answer_inputs(held_out, held_out_results),
    )
    return BenchmarkReport(
        candidate=candidate,
        dataset_id=dataset_id,
        strategy=strategy,
        tuning_digest=calibration.tuning_digest,
        held_out_digest=calibration.held_out_digest,
        metrics=metrics(cases, results),
        calibration=calibration,
    )


def metrics(
    cases: Sequence[EvaluationCase], results: Sequence[RetrievalResult]
) -> RetrievalMetrics:
    if len(cases) != len(results):
        raise ValueError("cases and results must have the same length")
    rows = tuple(_metric_row(case, result) for case, result in zip(cases, results))
    count = len(rows)
    average = lambda key: round(sum(row[key] for row in rows) / count, 6) if count else 0.0
    grouped: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        grouped[f"category:{row['category'] if row['category'] else 'unknown'}"].append(row)
        grouped[f"section:{row['section'] if row['section'] else 'unknown'}"].append(row)
    slices = {
        name: {
            "count": len(group),
            "hit_at_5": round(sum(row["hit"] for row in group) / len(group) * 100, 2),
            "recall_at_5": round(sum(row["recall"] for row in group) / len(group), 6),
        }
        for name, group in sorted(grouped.items())
    }
    return RetrievalMetrics(
        count=count,
        hit_at_5=round(average("hit") * 100, 2),
        recall_at_5=round(average("recall"), 6),
        mrr=round(average("rr"), 6),
        ndcg=round(average("ndcg"), 6),
        zero_hits=sum(row["hit"] == 0 for row in rows),
        slices=slices,
    )


def calibrate_threshold(
    cases: Sequence[EvaluationCase], results: Sequence[RetrievalResult]
) -> float:
    """Select the best threshold on tuning cases, with deterministic tie-breaking."""

    if len(cases) != len(results):
        raise ValueError("cases and results must have the same length")
    candidates = sorted(
        {PARTIAL_EVIDENCE_THRESHOLD, STRONG_EVIDENCE_THRESHOLD}
        | {round(score, 6) for result in results for score in result.scores}
    )
    scored = [
        (_refusal_accuracy(cases, results, threshold), -threshold, threshold)
        for threshold in candidates
    ]
    return max(scored)[2] if scored else PARTIAL_EVIDENCE_THRESHOLD


def split_digest(cases: Sequence[EvaluationCase]) -> str:
    payload = [case.model_dump(mode="json") for case in cases]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metric_row(case: EvaluationCase, result: RetrievalResult) -> dict[str, float | int | str]:
    retrieved = result.ordered_chunk_ids[:MAX_EVIDENCE_PASSAGES]
    relevant = set(case.relevant_chunk_ids)
    hits = [index for index, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant]
    hit = int(bool(hits))
    recall = len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0
    rr = 1 / hits[0] if hits else 0.0
    ideal = min(len(relevant), MAX_EVIDENCE_PASSAGES)
    dcg = sum(1 / log2(index + 1) for index in hits)
    idcg = sum(1 / log2(index + 1) for index in range(1, ideal + 1))
    return {
        "hit": hit,
        "recall": recall,
        "rr": rr,
        "ndcg": dcg / idcg if idcg else 0.0,
        "category": case.expected_outcome,
        "section": result.ordered_chunk_ids[0].split(":", 1)[0] if result.ordered_chunk_ids else "",
    }


def _retrieve(retriever: RetrievalRunner | Callable[[EvaluationCase], RetrievalResult], case: EvaluationCase) -> RetrievalResult:
    result = retriever.retrieve(case) if hasattr(retriever, "retrieve") else retriever(case)
    if len(result.ordered_chunk_ids) > MAX_EVIDENCE_PASSAGES:
        raise ValueError("retrieval result exceeds the five-passage cap")
    if result.stage_status != "succeeded":
        return result
    if result.score_semantics == "distance":
        scores = [1 - score for score in result.scores]
    else:
        scores = list(result.scores)
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("dense retrieval scores must be within [0, 1]")
    ranked = sorted(
        zip(result.ordered_chunk_ids, scores), key=lambda item: (-item[1], item[0])
    )
    return result.model_copy(
        update={
            "ordered_chunk_ids": [chunk_id for chunk_id, _ in ranked],
            "scores": [score for _, score in ranked],
            "score_semantics": "cosine_similarity",
        }
    )


def _freeze_cases(cases: Iterable[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    frozen = tuple(cases)
    if len({case.case_id for case in frozen}) != len(frozen):
        raise ValueError("evaluation case IDs must be unique within a split")
    return frozen


def _validate_splits(tuning: Sequence[EvaluationCase], held_out: Sequence[EvaluationCase], evaluated: Sequence[EvaluationCase]) -> None:
    ids = {case.case_id for case in evaluated}
    overlap = ids & ({case.case_id for case in tuning} | {case.case_id for case in held_out})
    if overlap:
        raise ValueError(f"evaluation cases overlap tuning/held-out splits: {sorted(overlap)}")


def _validate_candidate(results: Sequence[RetrievalResult], candidate: CandidateIdentity) -> None:
    if any(result.candidate != candidate for result in results):
        raise ValueError("all retrieval results must use the requested candidate")


def _refusal_accuracy(cases: Sequence[EvaluationCase], results: Sequence[RetrievalResult], threshold: float) -> float:
    if not cases:
        return 0.0
    correct = 0
    for case, result in zip(cases, results):
        has_evidence = bool(result.scores and max(result.scores) >= threshold)
        expected_refusal = case.expected_outcome == "evidence_refusal"
        correct += has_evidence == (not expected_refusal)
    return correct / len(cases)


def _false_refusals(cases: Sequence[EvaluationCase], results: Sequence[RetrievalResult], threshold: float) -> int:
    return sum(
        case.expected_outcome != "evidence_refusal"
        and not result.scores
        or case.expected_outcome != "evidence_refusal"
        and result.scores
        and max(result.scores) < threshold
        for case, result in zip(cases, results)
    )


def _unsupported_answer_inputs(cases: Sequence[EvaluationCase], results: Sequence[RetrievalResult]) -> int:
    return sum(
        case.expected_outcome == "evidence_refusal" and bool(result.ordered_chunk_ids)
        for case, result in zip(cases, results)
    )


def _as_json(value: object) -> object:
    if hasattr(value, "__dict__"):
        return {key: _as_json(item) for key, item in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    return value
