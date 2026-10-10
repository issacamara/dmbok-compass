import asyncio

from model.fallback import FallbackAdapter
from model.fake import FakeAdapter
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest


def test_cross_provider_fallback_preserves_attempts() -> None:
    primary = FakeAdapter(
        {"outcome": "answer", "answer_text": "unused"},
        provider="vertex",
        model="vertex",
        failure=GenerationError(GenerationErrorCode.UNAVAILABLE, "down", retryable=True),
    )
    secondary = FakeAdapter(
        {"outcome": "answer", "answer_text": "Grounded"},
        provider="openrouter",
        model="openrouter",
    )

    result = asyncio.run(FallbackAdapter(primary, secondary).generate(GenerationRequest(prompt="x")))

    assert result.candidate.answer_text == "Grounded"
    assert [attempt.model for attempt in result.usage.attempts] == ["vertex", "openrouter"]
