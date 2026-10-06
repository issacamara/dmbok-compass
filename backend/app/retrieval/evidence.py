"""Deterministic evidence classification for retrieved corpus passages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import AggregateMetric, RetrievedPassage

EvidenceOutcome = Literal["strong", "partial", "absent"]

# Scores are normalized to [0, 1] by the retrieval service.  Keeping the
# thresholds here makes the policy explicit and lets evaluation record the
# exact values used for a run.
STRONG_EVIDENCE_THRESHOLD = 0.75
PARTIAL_EVIDENCE_THRESHOLD = 0.45
MAX_EVIDENCE_PASSAGES = 5


class EvidenceBundle(BaseModel):
    """The bounded, content-bearing evidence input for answer generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: EvidenceOutcome
    basis: str = Field(min_length=1, max_length=500)
    cited_chunk_ids: tuple[str, ...] = Field(max_length=MAX_EVIDENCE_PASSAGES)
    passages: tuple[RetrievedPassage, ...] = Field(max_length=MAX_EVIDENCE_PASSAGES)


@dataclass(frozen=True)
class RetrievalCase:
    """One answerable gold question used by the top-five retrieval metric."""

    relevant_chunk_ids: frozenset[str]
    retrieved_chunk_ids: tuple[str, ...]


def classify_evidence(
    passages: Iterable[RetrievedPassage],
    *,
    strong_threshold: float = STRONG_EVIDENCE_THRESHOLD,
    partial_threshold: float = PARTIAL_EVIDENCE_THRESHOLD,
) -> EvidenceBundle:
    """Filter, rank, and classify passages without model or corpus calls.

    Passages without a finite normalized score are excluded.  Ties are broken
    by chunk ID so repeated evaluation runs produce the same evidence bundle.
    The five highest-scoring passages at or above the partial threshold are
    retained; the top score determines strong, partial, or absent evidence.
    """

    _validate_thresholds(strong_threshold, partial_threshold)
    candidates = sorted(
        (
            passage
            for passage in passages
            if passage.relevance_score is not None
            and isfinite(passage.relevance_score)
            and 0 <= passage.relevance_score <= 1
            and passage.relevance_score >= partial_threshold
        ),
        key=lambda passage: (-passage.relevance_score, passage.chunk_id),
    )[:MAX_EVIDENCE_PASSAGES]

    top_score = candidates[0].relevance_score if candidates else None
    if top_score is None:
        outcome: EvidenceOutcome = "absent"
    elif top_score >= strong_threshold:
        outcome = "strong"
    else:
        outcome = "partial"

    score_text = "none" if top_score is None else f"{top_score:.4f}"
    basis = (
        f"top_score={score_text}; strong_threshold={strong_threshold:.4f}; "
        f"partial_threshold={partial_threshold:.4f}"
    )
    return EvidenceBundle(
        outcome=outcome,
        basis=basis,
        cited_chunk_ids=tuple(passage.chunk_id for passage in candidates),
        passages=tuple(candidates),
    )


def retrieval_success_metric(cases: Iterable[RetrievalCase]) -> AggregateMetric:
    """Calculate top-five retrieval success as a content-free aggregate metric."""

    cases = tuple(cases)
    numerator = sum(
        bool(case.relevant_chunk_ids.intersection(case.retrieved_chunk_ids[:MAX_EVIDENCE_PASSAGES]))
        for case in cases
    )
    denominator = len(cases)
    percentage = round(100 * numerator / denominator, 2) if denominator else 0.0
    return AggregateMetric(
        metric_name="retrieval_success",
        numerator=numerator,
        denominator=denominator,
        percentage=percentage,
    )


def _validate_thresholds(strong_threshold: float, partial_threshold: float) -> None:
    if not all(isfinite(value) and 0 <= value <= 1 for value in (strong_threshold, partial_threshold)):
        raise ValueError("evidence thresholds must be finite values between 0 and 1")
    if partial_threshold >= strong_threshold:
        raise ValueError("partial_threshold must be lower than strong_threshold")
