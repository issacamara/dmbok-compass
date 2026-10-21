from __future__ import annotations

from pathlib import Path
import json

import pytest

from app.contracts import CandidateIdentity, EvaluationCase, RetrievalResult
from app.retrieval.benchmark import evaluate, metrics, split_digest


FIXTURE = json.loads((Path(__file__).parent / "fixtures/architecture-v21.json").read_text())
CANDIDATE = CandidateIdentity(**FIXTURE["candidateIdentity"])


def case(case_id: str, expected: str = "answer", relevant: list[str] | None = None) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Question {case_id}",
        expected_outcome=expected,
        relevant_chunk_ids=relevant or ["chunk-a"],
    )


def result(ids: list[str], scores: list[float], semantics: str = "cosine_similarity") -> RetrievalResult:
    return RetrievalResult(
        candidate=CANDIDATE,
        ordered_chunk_ids=ids,
        scores=scores,
        score_semantics=semantics,
        strategy="dense",
        stage_status="succeeded",
        search_ready=True,
    )


def test_metrics_use_top_five_denominators_and_reciprocal_rank() -> None:
    cases = [case("one", relevant=["chunk-a", "chunk-b"]), case("two", relevant=["missing"])]
    report = metrics(cases, [result(["chunk-a", "other"], [0.9, 0.4]), result(["other"], [0.8])])

    assert report.count == 2
    assert report.hit_at_5 == 50.0
    assert report.recall_at_5 == 0.25
    assert report.mrr == 0.5
    assert report.zero_hits == 1


def test_distance_scores_are_converted_and_ties_are_deterministic() -> None:
    item = case("one", relevant=["chunk-a"])
    report = evaluate(
        [item],
        lambda _: result(["chunk-b", "chunk-a"], [0.2, 0.2], "distance"),
        candidate=CANDIDATE,
        dataset_id="fixture-v1",
    )

    assert report.metrics.hit_at_5 == 100.0
    assert report.strategy == "dense-v1"


def test_calibration_records_split_digests_and_does_not_leak_held_out() -> None:
    tuning = [case("tuning", expected="evidence_refusal", relevant=[])]
    held_out = [case("held-out", expected="evidence_refusal", relevant=[])]
    report = evaluate(
        [],
        lambda item: result(["chunk-a"], [0.9]),
        candidate=CANDIDATE,
        dataset_id="reviewed-v1",
        tuning_cases=tuning,
        held_out_cases=held_out,
    )

    assert report.tuning_digest == split_digest(tuning)
    assert report.held_out_digest == split_digest(held_out)
    assert report.tuning_digest != report.held_out_digest
    assert report.calibration.policy_version.startswith("dense-evidence-")


def test_splits_cannot_overlap_evaluated_cases() -> None:
    item = case("same")
    with pytest.raises(ValueError, match="overlap"):
        evaluate(
            [item],
            lambda _: result(["chunk-a"], [0.9]),
            candidate=CANDIDATE,
            dataset_id="fixture-v1",
            held_out_cases=[item],
        )
