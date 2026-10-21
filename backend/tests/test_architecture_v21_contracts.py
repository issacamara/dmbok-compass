import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    AnswerOutcome,
    ApiError,
    CandidateIdentity,
    CitationEvidence,
    EvaluationCase,
    OPERATIONAL_ERROR_STATUS,
    QuotaStatus,
    RetrievalResult,
    RetrievalTrace,
    RuntimeSettings,
    RunItemResult,
)


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "architecture-v21.json").read_text()
)


def candidate() -> CandidateIdentity:
    return CandidateIdentity(**FIXTURE["candidateIdentity"])


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at=datetime.now(timezone.utc),
    )


def trace() -> RetrievalTrace:
    return RetrievalTrace(retrieved_passages=[], selected_model="test-model")


def test_fixture_validates_the_five_v21_boundaries() -> None:
    settings = RuntimeSettings(**FIXTURE["runtimeSettings"])
    case = EvaluationCase(**FIXTURE["evaluationCase"])
    retrieval = RetrievalResult(candidate=candidate(), **{key: FIXTURE["retrievalResult"][key] for key in (
        "ordered_chunk_ids", "scores", "score_semantics", "strategy", "stage_status", "search_ready",
    )})
    outcome = AnswerOutcome(
        **{key: FIXTURE["answerOutcome"][key] for key in ("outcome", "answer_text", "claims", "evidence")},
        trace=trace(),
        quota=quota(),
    )

    assert settings.secret_version == "openrouter-key-v3"
    result = RunItemResult(candidate=candidate(), **FIXTURE["runItemResult"])
    assert case.case_id == result.case_id
    assert retrieval.ordered_chunk_ids == ["chunk-1", "chunk-2"]
    assert outcome.outcome == "evidence_refusal"


def test_retrieval_result_keeps_ordered_ids_and_scores_aligned() -> None:
    with pytest.raises(ValidationError, match="align"):
        RetrievalResult(
            candidate=candidate(),
            ordered_chunk_ids=["chunk-1"],
            scores=[],
            score_semantics="cosine_similarity",
            strategy="dense",
            stage_status="succeeded",
        )

    with pytest.raises(ValidationError, match="search_ready"):
        RetrievalResult(
            candidate=candidate(),
            ordered_chunk_ids=[],
            scores=[],
            score_semantics="cosine_similarity",
            strategy="dense",
            stage_status="failed",
            search_ready=True,
        )


def test_outcome_statuses_keep_refusal_distinct_from_operational_failure() -> None:
    assert OPERATIONAL_ERROR_STATUS == {"invalid_output": 502, "unavailable": 503, "deadline": 504}
    refusal = AnswerOutcome(
        outcome="evidence_refusal", trace=trace(), quota=quota(), answer_text="No evidence."
    )
    assert refusal.error is None

    with pytest.raises(ValidationError, match="safe error"):
        AnswerOutcome(outcome="operational_error", trace=trace(), quota=quota())
    with pytest.raises(ValidationError, match="completed outcome"):
        AnswerOutcome(
            outcome="evidence_refusal",
            trace=trace(),
            quota=quota(),
            error=ApiError(code="unavailable", message="Try again", status=503),
        )


def test_citation_evidence_and_runtime_settings_are_bounded_and_reference_only() -> None:
    evidence = CitationEvidence(
        citation_id="citation-1",
        claim_id="claim-1",
        chunk_id="chunk-1",
        page=2,
        section="Retention",
        excerpt="Bounded source excerpt.",
    )
    assert evidence.page == 2
    with pytest.raises(ValidationError):
        RuntimeSettings(
            project="p",
            location="l",
            provider="openrouter",
            primary_model="m",
            secret_version="raw-secret-value-that-is-not-a-version-reference",
            unexpected="secret",  # type: ignore[call-arg]
        )
