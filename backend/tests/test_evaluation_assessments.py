from app.contracts import GoldAnnotation, ReviewedSemanticAssessment
from app.evaluation.assessments import response_fingerprint, score_assessment


def assessment(*, answer: str, cited: set[str], outcome: str = "answer", **overrides: object) -> ReviewedSemanticAssessment:
    fields: dict[str, object] = {
        "response_sha256": response_fingerprint(outcome=outcome, answer_text=answer, cited_chunk_ids=cited),
        "scorer_version": "evaluation-gates-v1",
        "claim_support": True,
        "claim_citation_support": True,
        "rubric_passed": True,
        "expected_refusal": False,
    }
    fields.update(overrides)
    return ReviewedSemanticAssessment(**fields)


def annotation(review: ReviewedSemanticAssessment | None) -> GoldAnnotation:
    return GoldAnnotation(
        item_id="item-1",
        expected_answer_or_rubric="Reviewed rubric",
        relevant_chunk_ids=["chunk-1"],
        review_status="approved",
        semantic_assessment=review,
    )


def test_hallucinated_cited_nonempty_response_fails_reviewed_semantic_metrics() -> None:
    reviewed = assessment(
        answer="Incorrect but confident answer.",
        cited={"chunk-1"},
        claim_support=False,
        claim_citation_support=False,
        rubric_passed=False,
    )

    score = score_assessment(
        annotation(reviewed), outcome="answer", answer_text="Incorrect but confident answer.",
        retrieved_chunk_ids={"chunk-1"}, cited_chunk_ids={"chunk-1"}, scorer_version="evaluation-gates-v1",
    )

    assert score.retrieval_success is True
    assert (score.grounded, score.citation_correct, score.answer_quality) == (False, False, False)


def test_wrong_claim_support_citation_and_unreviewed_response_fail_closed() -> None:
    reviewed = assessment(answer="Correct words, wrong support.", cited={"chunk-2"}, claim_citation_support=False)

    wrong_support = score_assessment(
        annotation(reviewed), outcome="answer", answer_text="Correct words, wrong support.",
        retrieved_chunk_ids={"chunk-1"}, cited_chunk_ids={"chunk-2"}, scorer_version="evaluation-gates-v1",
    )
    unreviewed = score_assessment(
        annotation(None), outcome="answer", answer_text="Anything", retrieved_chunk_ids={"chunk-1"},
        cited_chunk_ids={"chunk-1"}, scorer_version="evaluation-gates-v1",
    )

    assert wrong_support.citation_correct is False
    assert unreviewed.applicable_metrics == frozenset({
        "retrieval_success", "grounded_claims", "citation_correctness", "answer_quality", "refusal_correctness",
    })
    assert not any((unreviewed.retrieval_success, unreviewed.grounded, unreviewed.citation_correct, unreviewed.answer_quality, unreviewed.refusal_correct))


def test_refusal_and_partial_cases_have_metric_specific_applicability() -> None:
    refusal = ReviewedSemanticAssessment(
        response_sha256="a" * 64, scorer_version="evaluation-gates-v1", expected_refusal=True,
    )
    partial = assessment(answer="A qualified, reviewed partial answer.", cited={"chunk-1"}, outcome="qualified")

    refusal_score = score_assessment(
        GoldAnnotation(item_id="refusal", expected_answer_or_rubric="Refuse", relevant_chunk_ids=[], review_status="approved", semantic_assessment=refusal),
        outcome="refusal", answer_text=None, retrieved_chunk_ids=set(), cited_chunk_ids=set(), scorer_version="evaluation-gates-v1",
    )
    partial_score = score_assessment(
        annotation(partial), outcome="qualified", answer_text="A qualified, reviewed partial answer.",
        retrieved_chunk_ids={"chunk-1"}, cited_chunk_ids={"chunk-1"}, scorer_version="evaluation-gates-v1",
    )

    assert refusal_score.applicable_metrics == frozenset({"refusal_correctness"})
    assert refusal_score.refusal_correct is True
    assert partial_score.applicable_metrics == frozenset({"retrieval_success", "grounded_claims", "citation_correctness", "answer_quality"})
