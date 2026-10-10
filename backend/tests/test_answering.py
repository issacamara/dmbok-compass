from __future__ import annotations

import asyncio

import pytest

from app.answering import GroundedAnswerPolicy, RetrievedAnswerService, classify_evidence
from app.contracts import CorpusVersion, QuotaStatus, RetrievedPassage
from model.fake import FakeAdapter
from model.protocol import GenerationCandidate


def quota() -> QuotaStatus:
    from datetime import datetime, timezone

    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
    )


def passage(chunk_id: str, score: float | None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=4,
        section="Governance",
        excerpt=f"Stored excerpt {chunk_id}",
        relevance_score=score,
    )


class FakeRetriever:
    def __init__(self, passages: tuple[RetrievedPassage, ...]) -> None:
        self.passages = passages
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, question: str, *, active_corpus_version_id: str, limit: int = 5):
        self.calls.append((question, active_corpus_version_id, limit))
        return self.passages


class FakeCorpusStore:
    def __init__(self, active: CorpusVersion | None) -> None:
        self._active = active

    def active(self):
        return self._active


def active_version() -> CorpusVersion:
    from datetime import datetime, timezone

    return CorpusVersion(
        version_id="v1",
        source_uri="gs://bucket/dmbok.pdf",
        embedding_model="gemini-embedding-001",
        status="active",
        created_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("score", "outcome"),
    [(0.75, "strong"), (0.45, "partial"), (0.44, "absent"), (None, "absent")],
)
def test_classify_evidence_boundaries(score: float | None, outcome: str) -> None:
    assert classify_evidence([passage("chunk-1", score)]).outcome == outcome


def test_retrieved_answer_service_uses_active_version_and_top_five_limit() -> None:
    import asyncio

    retriever = FakeRetriever((passage("chunk-1", 0.9),))
    policy = GroundedAnswerPolicy(
        FakeAdapter(
            {
                "outcome": "answer",
                "answer_text": "Grounded.",
                "citations": [{"citation_id": "chunk-1", "page": 1, "section": "x", "excerpt": "y"}],
            }
        )
    )

    response = asyncio.run(
        RetrievedAnswerService(retriever, FakeCorpusStore(active_version()), policy).answer(
            "What is governance?", quota()
        )
    )

    assert response.outcome == "answer"
    assert retriever.calls == [("What is governance?", "v1", 5)]


def test_retrieved_answer_service_refuses_without_active_corpus() -> None:
    import asyncio

    policy = GroundedAnswerPolicy(FakeAdapter({"outcome": "refusal", "answer_text": "unused"}))
    response = asyncio.run(
        RetrievedAnswerService(FakeRetriever(()), FakeCorpusStore(None), policy).answer(
            "What is governance?", quota()
        )
    )

    assert response.outcome == "refusal"
    assert response.error is not None
    assert response.error.code == "active_corpus_unavailable"


def test_absent_evidence_refuses_without_calling_model() -> None:
    adapter = FakeAdapter({"outcome": "answer", "answer_text": "Should not be used."})
    policy = GroundedAnswerPolicy(adapter)

    response = asyncio.run(policy.answer("What is governance?", [passage("chunk-1", 0.2)], quota()))

    assert response.outcome == "refusal"
    assert response.answer_text
    assert response.citations == []


def test_strong_answer_uses_stored_citation_provenance_and_labels_synthesis() -> None:
    adapter = FakeAdapter(
        GenerationCandidate(
            outcome="answer",
            answer_text="Governance coordinates decisions.",
            synthesis=False,
            citations=[
                {"citation_id": "chunk-1", "page": 999, "section": "fake", "excerpt": "fake"},
                {"citation_id": "chunk-2", "page": 999, "section": "fake", "excerpt": "fake"},
            ],
        ),
        model="primary-model",
    )
    policy = GroundedAnswerPolicy(adapter)

    response = asyncio.run(policy.answer(
        "What is governance?",
        [passage("chunk-2", 0.8), passage("chunk-1", 0.9)],
        quota(),
    ))

    assert response.outcome == "answer"
    assert response.synthesis is True
    assert response.trace.selected_model == "primary-model"
    assert [(item.page, item.section, item.excerpt) for item in response.citations] == [
        (4, "Governance", "Stored excerpt chunk-1"),
        (4, "Governance", "Stored excerpt chunk-2"),
    ]
    assert response.trace.timings_ms["generation_ms"] >= 0
    assert response.trace.timings_ms["total_ms"] >= response.trace.timings_ms["generation_ms"]


def test_partial_evidence_forces_qualified_outcome() -> None:
    adapter = FakeAdapter(
        {
            "outcome": "answer",
            "answer_text": "A limited answer.",
            "citations": [{"citation_id": "chunk-1", "page": 1, "section": "x", "excerpt": "y"}],
        }
    )
    response = asyncio.run(GroundedAnswerPolicy(adapter).answer(
        "Explain this.", [passage("chunk-1", 0.5)], quota()
    ))

    assert response.outcome == "qualified"


def test_unmapped_citation_is_rejected() -> None:
    adapter = FakeAdapter(
        {
            "outcome": "answer",
            "answer_text": "Unsupported.",
            "citations": [{"citation_id": "not-retrieved", "page": 1, "section": "x", "excerpt": "y"}],
        }
    )

    response = asyncio.run(GroundedAnswerPolicy(adapter).answer(
        "Explain this.", [passage("chunk-1", 0.9)], quota()
    ))

    assert response.outcome == "refusal"
    assert response.error is not None
    assert response.error.code == "grounding_policy_rejected"


def test_retrieved_service_includes_retrieval_timing_in_ephemeral_trace() -> None:
    response = asyncio.run(
        RetrievedAnswerService(
            FakeRetriever((passage("chunk-1", 0.9),)),
            FakeCorpusStore(active_version()),
            GroundedAnswerPolicy(
                FakeAdapter(
                    {
                        "outcome": "answer",
                        "answer_text": "Grounded.",
                        "citations": [{"citation_id": "chunk-1"}],
                    }
                )
            ),
        ).answer("What is governance?", quota())
    )

    assert response.trace.timings_ms["retrieval_ms"] >= 0
    assert response.trace.timings_ms["total_ms"] >= response.trace.timings_ms["retrieval_ms"]


def test_provider_failure_is_safe_refusal() -> None:
    from model.protocol import GenerationError, GenerationErrorCode
    from app.contracts.api import ModelAttempt

    adapter = FakeAdapter(
        {"outcome": "answer", "answer_text": "unused"},
        failure=GenerationError(
            GenerationErrorCode.UNAVAILABLE, "Provider unavailable.", retryable=True,
            attempts=(
                ModelAttempt(model="primary", outcome="unavailable", status_code=404),
                ModelAttempt(model="fallback", outcome="unavailable", status_code=503),
            ),
        ),
    )

    response = asyncio.run(GroundedAnswerPolicy(adapter).answer(
        "Explain this.", [passage("chunk-1", 0.9)], quota()
    ))

    assert response.outcome == "refusal"
    assert response.error is not None
    assert response.error.retryable is True
    assert response.trace.selected_model == "fallback"
    assert [attempt.status_code for attempt in response.trace.model_attempts] == [404, 503]
