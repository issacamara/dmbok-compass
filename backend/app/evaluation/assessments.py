"""Fail-closed semantic scoring from frozen human-reviewed assessments."""

from __future__ import annotations

from hashlib import sha256
import json
from dataclasses import dataclass

from app.contracts import GoldAnnotation, ReviewedSemanticAssessment


SEMANTIC_METRICS = frozenset({
    "retrieval_success", "grounded_claims", "citation_correctness", "answer_quality", "refusal_correctness",
})


@dataclass(frozen=True)
class SemanticScore:
    retrieval_success: bool
    grounded: bool
    citation_correct: bool
    answer_quality: bool
    refusal_correct: bool
    applicable_metrics: frozenset[str]


def response_fingerprint(*, outcome: str, answer_text: str | None, cited_chunk_ids: set[str]) -> str:
    """Hash only the ephemeral response material that was reviewed by a human."""

    payload = {
        "answer_text": answer_text or "",
        "cited_chunk_ids": sorted(cited_chunk_ids),
        "outcome": outcome,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_assessment(
    annotation: GoldAnnotation | None,
    *,
    outcome: str,
    answer_text: str | None,
    retrieved_chunk_ids: set[str],
    cited_chunk_ids: set[str],
    scorer_version: str,
    operational_error: bool = False,
) -> SemanticScore:
    """Return independent retrieval and human-reviewed semantic evidence.

    Retrieval is scored from approved support IDs and ranked retrieval IDs
    before validating the generated response.  Missing, non-approved, stale,
    or response-mismatched assessments still fail the semantic gates without
    turning a correct retrieval into a retrieval miss.
    """

    retrieval_success = _retrieval_success(annotation, retrieved_chunk_ids)
    assessment = annotation.semantic_assessment if annotation is not None else None
    if not _is_valid(assessment, annotation, outcome, answer_text, cited_chunk_ids, scorer_version):
        return SemanticScore(retrieval_success, False, False, False, False, SEMANTIC_METRICS)

    assert annotation is not None
    assert assessment is not None
    expected = set(annotation.relevant_chunk_ids)
    if assessment.expected_refusal:
        return SemanticScore(
            retrieval_success=False,
            grounded=False,
            citation_correct=False,
            answer_quality=False,
            refusal_correct=outcome == "refusal" and not operational_error,
            applicable_metrics=frozenset({"refusal_correctness"}),
        )

    # Answerable and partial cases need reviewed claim/citation/rubric evidence.
    applicable = {"grounded_claims", "citation_correctness", "answer_quality"}
    if expected:
        applicable.add("retrieval_success")
    return SemanticScore(
        retrieval_success=retrieval_success,
        grounded=assessment.claim_support is True,
        citation_correct=assessment.claim_citation_support is True,
        answer_quality=assessment.rubric_passed is True,
        refusal_correct=False,
        applicable_metrics=frozenset(applicable),
    )


def _retrieval_success(annotation: GoldAnnotation | None, retrieved_chunk_ids: set[str]) -> bool:
    """Score retrieval from reviewed support IDs, independently of generation."""

    if annotation is None or annotation.review_status != "approved":
        return False
    expected = set(annotation.relevant_chunk_ids)
    return bool(expected and retrieved_chunk_ids & expected)


def _is_valid(
    assessment: ReviewedSemanticAssessment | None,
    annotation: GoldAnnotation | None,
    outcome: str,
    answer_text: str | None,
    cited_chunk_ids: set[str],
    scorer_version: str,
) -> bool:
    if annotation is None or annotation.review_status != "approved" or assessment is None:
        return False
    if assessment.scorer_version != scorer_version or assessment.expected_refusal is None:
        return False
    if assessment.expected_refusal:
        return True
    if None in {assessment.claim_support, assessment.claim_citation_support, assessment.rubric_passed}:
        return False
    return assessment.response_sha256 == response_fingerprint(
        outcome=outcome, answer_text=answer_text, cited_chunk_ids=cited_chunk_ids
    )
