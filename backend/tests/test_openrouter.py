import asyncio
import json

import httpx
import pytest

from model.openrouter import OpenRouterAdapter
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest, TimeoutBudget


def request(**kwargs: object) -> GenerationRequest:
    return GenerationRequest(prompt="Answer from supplied evidence.", **kwargs)


def response_payload(answer: str = "Grounded answer.") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "outcome": "answer",
                            "answer_text": answer,
                            "synthesis": False,
                            "citations": [],
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }


def test_openrouter_parses_candidate_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "google/gemma-2-27b-it"
        return httpx.Response(200, json=response_payload())

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request()))

    assert result.candidate.answer_text == "Grounded answer."
    assert result.usage.model_dump() == {
        "provider": "openrouter",
        "model": "google/gemma-2-27b-it",
        "input_tokens": 10,
        "output_tokens": 6,
        "total_tokens": 16,
    }


def test_openrouter_uses_free_fallback_for_rate_limit() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        if len(models) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=response_payload("Fallback answer."))

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request()))

    assert models == ["google/gemma-2-27b-it", "nvidia/nemotron-3.5-lightning:free"]
    assert result.candidate.answer_text == "Fallback answer."
    assert result.usage.model == "nvidia/nemotron-3.5-lightning:free"


def test_openrouter_does_not_fallback_on_authentication_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))

    with pytest.raises(GenerationError) as raised:
        asyncio.run(adapter.generate(request()))

    assert raised.value.code is GenerationErrorCode.AUTHENTICATION_FAILED
    assert calls == 1


def test_openrouter_respects_total_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, json=response_payload())

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))

    with pytest.raises(GenerationError) as raised:
        asyncio.run(adapter.generate(request(timeout=TimeoutBudget(total_ms=1))))

    assert raised.value.code is GenerationErrorCode.TIMEOUT
