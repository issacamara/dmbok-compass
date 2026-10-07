import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import AnswerResponse, DocumentChunk, TelemetryEvent, validate_telemetry_payload
from app.contracts.api import Citation, QuotaStatus, RetrievalTrace
from app.contracts.telemetry import TELEMETRY_FIELDS


FIXTURE = json.loads(
    (Path(__file__).parents[2] / "documents/contracts/api-contract-fixture.json").read_text()
)


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


def test_answer_contract_supports_all_outcomes_and_requires_safe_shape() -> None:
    answered = AnswerResponse(
        outcome="answer",
        answer_text="Grounded answer.",
        citations=[Citation(citation_id="c1", page=1, section="Intro", excerpt="Evidence")],
        trace=trace(),
        quota=quota(),
    )
    assert answered.outcome == "answer"

    qualified = AnswerResponse(outcome="qualified", answer_text="Limited answer.", trace=trace(), quota=quota())
    refused = AnswerResponse(outcome="refusal", answer_text="No relevant evidence.", trace=trace(), quota=quota())
    assert {qualified.outcome, refused.outcome} == {"qualified", "refusal"}

    with pytest.raises(ValidationError):
        AnswerResponse(outcome="answer", trace=trace(), quota=quota())


def test_document_chunk_requires_exactly_768_dimensions() -> None:
    valid = DocumentChunk(
        chunk_id="v1:1:0",
        corpus_version_id="v1",
        page=1,
        section="Intro",
        chunk_ordinal=0,
        content_hash="a" * 64,
        text="Evidence",
        embedding=[0.0] * FIXTURE["embeddingDimensions"],
    )
    assert len(valid.embedding) == 768

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id=valid.chunk_id,
            corpus_version_id=valid.corpus_version_id,
            page=valid.page,
            section=valid.section,
            chunk_ordinal=valid.chunk_ordinal,
            content_hash=valid.content_hash,
            text=valid.text,
            embedding=[0.0] * 767,
        )


def test_telemetry_is_allowlisted_and_rejects_interaction_content() -> None:
    event = TelemetryEvent(outcome="answer", duration_ms=12.5, request_count=1)
    assert set(event.model_dump(exclude_none=True)) <= set(TELEMETRY_FIELDS)

    with pytest.raises(ValueError):
        validate_telemetry_payload({"nested": {"question": "secret input"}})


def test_answer_response_cannot_be_forwarded_as_telemetry_payload() -> None:
    response = AnswerResponse(
        outcome="answer",
        answer_text="Do not persist this.",
        trace=trace(),
        quota=quota(),
    )

    with pytest.raises(ValueError, match="answer_text"):
        validate_telemetry_payload(response.model_dump())
    with pytest.raises(ValidationError):
        TelemetryEvent(question="secret input")  # type: ignore[call-arg]


def test_cross_language_fixture_matches_backend_contract_values() -> None:
    assert list(TELEMETRY_FIELDS) == FIXTURE["telemetryFields"]
    assert set(FIXTURE["outcomes"]) == {"answer", "qualified", "refusal"}
