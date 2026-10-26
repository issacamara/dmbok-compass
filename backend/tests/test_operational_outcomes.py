import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.contracts import AnswerResponse, ApiError, QuotaStatus, QuestionRequest, RetrievalTrace
from app.main import answer_question


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=2,
        user_limit=100,
        global_used=2,
        global_limit=100,
        resets_at=datetime.now(timezone.utc),
    )


class Answerer:
    def __init__(self, response: AnswerResponse) -> None:
        self.response = response

    async def answer(self, question: str, quota: QuotaStatus) -> AnswerResponse:
        return self.response


class Telemetry:
    def emit(self, event: object) -> None:
        pass


def operational_response(status: int) -> AnswerResponse:
    return AnswerResponse(
        outcome="operational_error",
        trace=RetrievalTrace(retrieved_passages=[], selected_model="test-model"),
        quota=quota(),
        error=ApiError(code="answer_service_failure", message="safe failure", status=status),
    )


@pytest.mark.parametrize("status", [502, 503, 504])
def test_operational_outcomes_are_non_2xx_and_preserve_safe_metadata(status: int) -> None:
    response = asyncio.run(answer_question(
        QuestionRequest(question="What is governance?"),
        quota=quota(),
        answerer=Answerer(operational_response(status)),
        telemetry=Telemetry(),
    ))

    assert response.status_code == status
    payload = json.loads(response.body)
    assert payload["outcome"] == "operational_error"
    assert payload["error"]["status"] == status
    assert payload["trace"]["selected_model"] == "test-model"
    assert payload["quota"]["user_used"] == 2
    assert payload["detail"] == payload["error"]


def test_completed_evidence_refusal_remains_a_200_response() -> None:
    response = asyncio.run(answer_question(
        QuestionRequest(question="What is governance?"),
        quota=quota(),
        answerer=Answerer(AnswerResponse(
            outcome="refusal",
            answer_text="I could not find enough relevant evidence.",
            trace=RetrievalTrace(retrieved_passages=[], selected_model="test-model"),
            quota=quota(),
        )),
        telemetry=Telemetry(),
    ))

    assert isinstance(response, AnswerResponse)
    assert response.outcome == "refusal"
