import asyncio
import json

import httpx
import pytest

from model.openrouter import OpenRouterAdapter
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest, TimeoutBudget


def request(**kwargs: object) -> GenerationRequest:
    return GenerationRequest(prompt="Answer only from supplied evidence.", **kwargs)


def response(answer: str = "Grounded answer.") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": json.dumps({
            "outcome": "answer", "answer_text": answer, "synthesis": False, "citations": []
        })}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_runnable_preserves_primary_result_and_safe_usage() -> None:
    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response())))
    result = asyncio.run(adapter.as_langchain_runnable().ainvoke(request()))

    assert result.candidate.answer_text == "Grounded answer."
    assert result.usage.model == "nvidia/nemotron-3.5-lightning:free"
    assert result.usage.model_dump(exclude={"attempts"}) == {
        "provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free",
        "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
    }


def test_runnable_preserves_eligible_fallback() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429) if calls == 1 else httpx.Response(200, json=response("Fallback answer."))

    result = asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).as_langchain_runnable().ainvoke(request()))
    assert result.candidate.answer_text == "Fallback answer."
    assert [attempt.outcome for attempt in result.usage.attempts] == ["rate_limited", "success"]


def test_runnable_does_not_fallback_for_authentication_failure() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    with pytest.raises(GenerationError, match="authentication") as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).as_langchain_runnable().ainvoke(request()))
    assert raised.value.code is GenerationErrorCode.AUTHENTICATION_FAILED
    assert calls == 1


def test_runnable_preserves_timeout_contract() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, json=response())

    with pytest.raises(GenerationError) as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).as_langchain_runnable().ainvoke(
            request(timeout=TimeoutBudget(total_ms=1))
        ))
    assert raised.value.code is GenerationErrorCode.TIMEOUT


def test_runnable_preserves_malformed_output_contract() -> None:
    payload = {"choices": [{"message": {"content": "not JSON"}}]}
    with pytest.raises(GenerationError) as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )).as_langchain_runnable().ainvoke(request()))
    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT


def test_runnable_rejects_content_bearing_callbacks() -> None:
    runnable = OpenRouterAdapter("test-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response()))).as_langchain_runnable()
    with pytest.raises(ValueError, match="callbacks"):
        asyncio.run(runnable.ainvoke(request(), {"callbacks": [object()]}))
