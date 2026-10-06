from __future__ import annotations

import pytest

from app.contracts import RetrievedPassage
from app.retrieval.evidence import (
    PARTIAL_EVIDENCE_THRESHOLD,
    STRONG_EVIDENCE_THRESHOLD,
    RetrievalCase,
    classify_evidence,
    retrieval_success_metric,
)


def passage(chunk_id: str, score: float | None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=1,
        section="Governance",
        excerpt=f"Excerpt {chunk_id}",
        relevance_score=score,
    )


@pytest.mark.parametrize(
    ("score", "outcome"),
    [
        (STRONG_EVIDENCE_THRESHOLD, "strong"),
        (PARTIAL_EVIDENCE_THRESHOLD, "partial"),
        (PARTIAL_EVIDENCE_THRESHOLD - 0.01, "absent"),
        (None, "absent"),
    ],
)
def test_classifies_evidence_at_explicit_boundaries(score: float | None, outcome: str) -> None:
    bundle = classify_evidence([passage("chunk-1", score)])

    assert bundle.outcome == outcome
    assert bundle.basis
    assert len(bundle.cited_chunk_ids) <= 5
    assert bundle.cited_chunk_ids == (("chunk-1",) if outcome != "absent" else ())


def test_filters_invalid_scores_sorts_ties_deterministically_and_caps_at_five() -> None:
    passages = [
        passage(f"chunk-{index}", 0.9 if index % 2 else 0.8)
        for index in range(7)
    ]
    passages.extend([passage("invalid", 1.1), passage("missing", None)])

    bundle = classify_evidence(passages)

    assert bundle.outcome == "strong"
    assert bundle.cited_chunk_ids == (
        "chunk-1",
        "chunk-3",
        "chunk-5",
        "chunk-0",
        "chunk-2",
    )


def test_retrieval_success_uses_top_five_and_reports_numerator_denominator() -> None:
    metric = retrieval_success_metric(
        [
            RetrievalCase(frozenset({"hit-1"}), ("hit-1",)),
            RetrievalCase(frozenset({"hit-2"}), ("other", "hit-2")),
            RetrievalCase(frozenset({"outside"}), ("a", "b", "c", "d", "e", "outside")),
            RetrievalCase(frozenset({"missing"}), ("a", "b")),
        ]
    )

    assert metric.metric_name == "retrieval_success"
    assert metric.numerator == 2
    assert metric.denominator == 4
    assert metric.percentage == 50.0


def test_retrieval_success_empty_fixture_is_zero() -> None:
    assert retrieval_success_metric([]).model_dump() == {
        "metric_name": "retrieval_success",
        "numerator": 0,
        "denominator": 0,
        "percentage": 0.0,
    }


def test_thresholds_must_be_ordered_and_bounded() -> None:
    with pytest.raises(ValueError, match="lower than"):
        classify_evidence([passage("chunk-1", 0.8)], strong_threshold=0.5, partial_threshold=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        classify_evidence([passage("chunk-1", 0.8)], strong_threshold=1.1)
