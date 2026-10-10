import asyncio

import pytest

from model.fake import FakeAdapter
from model.protocol import (
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    TimeoutBudget,
    parse_candidate,
)


def request(**kwargs: object) -> GenerationRequest:
    return GenerationRequest(prompt="Answer from the supplied evidence.", **kwargs)


def test_fake_adapter_returns_valid_candidate_and_safe_usage() -> None:
    adapter = FakeAdapter(
        {"outcome": "answer", "answer_text": "Grounded answer."},
        input_tokens=4,
        output_tokens=3,
    )

    result = asyncio.run(adapter.generate(request()))

    assert result.candidate.answer_text == "Grounded answer."
    assert result.usage.model_dump() == {
        "provider": "fake",
        "model": "fake-model",
        "input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 7,
        "attempts": [],
    }


def test_invalid_structured_output_is_normalized() -> None:
    with pytest.raises(GenerationError) as error:
        parse_candidate({"outcome": "answer"})

    assert error.value.code is GenerationErrorCode.INVALID_OUTPUT
    assert error.value.message == "The model returned an incomplete structured candidate."


def test_fake_adapter_normalizes_configured_failure() -> None:
    adapter = FakeAdapter(
        {"outcome": "refusal", "answer_text": "No evidence."},
        failure=GenerationError(GenerationErrorCode.RATE_LIMITED, "Try again later.", retryable=True),
    )

    with pytest.raises(GenerationError) as error:
        asyncio.run(adapter.generate(request()))

    assert error.value.as_dict() == {
        "code": "rate_limited",
        "message": "Try again later.",
        "retryable": True,
    }


def test_fake_adapter_enforces_timeout_budget() -> None:
    adapter = FakeAdapter({"outcome": "refusal", "answer_text": "No evidence."}, delay_ms=20)

    with pytest.raises(GenerationError) as error:
        asyncio.run(adapter.generate(request(timeout=TimeoutBudget(total_ms=1))))

    assert error.value.code is GenerationErrorCode.TIMEOUT
    assert error.value.retryable is True
