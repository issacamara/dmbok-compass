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
        assert body["model"] == "nvidia/nemotron-3.5-lightning:free"
        return httpx.Response(200, json=response_payload())

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request()))

    assert result.candidate.answer_text == "Grounded answer."
    assert result.usage.model_dump() == {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3.5-lightning:free",
        "input_tokens": 10,
        "output_tokens": 6,
        "total_tokens": 16,
        "attempts": [{"model": "nvidia/nemotron-3.5-lightning:free", "outcome": "success", "status_code": 200}],
    }


def test_openrouter_requests_strict_candidate_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json=response_payload())

    asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))


@pytest.mark.parametrize("outcome,answer_text,citations", [
    ("answer", "Grounded answer.", []),
    ("qualified", "Partially grounded answer.", []),
    ("refusal", None, []),
])
def test_openrouter_accepts_each_candidate_outcome(
    outcome: str, answer_text: str | None, citations: list[object]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["choices"][0]["message"]["content"] = json.dumps({
            "outcome": outcome,
            "answer_text": answer_text,
            "synthesis": False,
            "citations": citations,
        })
        return httpx.Response(200, json=payload)

    result = asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))
    assert result.candidate.outcome == outcome
    assert result.candidate.answer_text == answer_text


@pytest.mark.parametrize("content", [
    "Reasoning only; no candidate JSON.",
    "prefix " + json.dumps({"outcome": "answer", "answer_text": "Grounded answer.", "synthesis": False, "citations": []}),
    "```json\n" + json.dumps({"outcome": "answer", "answer_text": "Grounded answer.", "synthesis": False, "citations": []}) + "\n```",
])
def test_openrouter_rejects_reasoning_or_non_object_json(content: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["choices"][0]["message"]["content"] = content
        return httpx.Response(200, json=payload)

    with pytest.raises(GenerationError) as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))
    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT


def test_openrouter_accepts_text_content_block_with_exact_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["choices"][0]["message"]["content"] = [{
            "type": "text",
            "text": payload["choices"][0]["message"]["content"],
        }]
        return httpx.Response(200, json=payload)

    result = asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))
    assert result.candidate.answer_text == "Grounded answer."


def test_openrouter_rejects_unknown_citation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["choices"][0]["message"]["content"] = json.dumps({
            "outcome": "answer", "answer_text": "Grounded answer.", "synthesis": False,
            "citations": [{"citation_id": "missing", "page": 1, "section": "S", "excerpt": "E"}],
        })
        return httpx.Response(200, json=payload)

    with pytest.raises(GenerationError) as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(
            request(passages=[{"passage_id": "p1", "page": 1, "section": "S", "text": "E"}])
        ))
    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT


def test_openrouter_uses_json_mode_when_route_does_not_support_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=response_payload())

    result = asyncio.run(OpenRouterAdapter(
        "test-key", json_schema_supported=False, transport=httpx.MockTransport(handler)
    ).generate(request()))
    assert result.candidate.answer_text == "Grounded answer."


def test_openrouter_retries_json_object_when_schema_is_rejected() -> None:
    formats: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response_format = json.loads(request.content)["response_format"]
        formats.append(response_format)
        if len(formats) == 1:
            return httpx.Response(400, json={"error": {"message": "json_schema unsupported"}})
        return httpx.Response(200, json=response_payload())

    result = asyncio.run(OpenRouterAdapter(
        "test-key", transport=httpx.MockTransport(handler)
    ).generate(request()))
    assert formats == [
        {"type": "json_schema", "json_schema": formats[0]["json_schema"]},
        {"type": "json_object"},
    ]
    assert result.candidate.answer_text == "Grounded answer."


def test_openrouter_uses_free_fallback_for_rate_limit() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        if len(models) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=response_payload("Fallback answer."))

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request()))

    assert models == ["nvidia/nemotron-3.5-lightning:free", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"]
    assert result.candidate.answer_text == "Fallback answer."
    assert result.usage.model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    assert [attempt.outcome for attempt in result.usage.attempts] == ["rate_limited", "success"]


def test_openrouter_reports_both_failed_routes_without_request_content() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return httpx.Response(404 if len(models) == 1 else 503)

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    with pytest.raises(GenerationError) as raised:
        asyncio.run(adapter.generate(request()))

    assert [(attempt.model, attempt.outcome, attempt.status_code) for attempt in raised.value.attempts] == [
        ("nvidia/nemotron-3.5-lightning:free", "unavailable", 404),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "unavailable", 503),
    ]
    assert "Answer from supplied evidence" not in str(raised.value.attempts)


def test_openrouter_requests_contract_citation_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert '"citation_id":string' in body["messages"][0]["content"]
        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning"] == {"exclude": True}
        return httpx.Response(200, json=response_payload())

    asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))


def test_openrouter_trace_output_excludes_reasoning_wrapper() -> None:
    reasoning_wrapped = "Reasoning that must not be shown.\n" + json.dumps({
        "outcome": "answer",
        "answer_text": "Grounded answer.",
        "synthesis": False,
        "citations": [],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["choices"][0]["message"]["content"] = reasoning_wrapped
        return httpx.Response(200, json=payload)

    result = asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler)).generate(request()))

    assert result.raw_output == reasoning_wrapped[reasoning_wrapped.index("{"):]
    assert "Reasoning that must not be shown" not in (result.raw_output or "")


def test_openrouter_uses_free_fallback_when_primary_model_is_missing() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        if len(models) == 1:
            return httpx.Response(404)
        return httpx.Response(200, json=response_payload("Fallback answer."))

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request()))

    assert models == ["nvidia/nemotron-3.5-lightning:free", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"]
    assert result.candidate.answer_text == "Fallback answer."


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


def test_openrouter_allows_fallback_after_primary_timeout() -> None:
    models: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        models.append(model)
        if len(models) == 1:
            await asyncio.sleep(0.02)
        return httpx.Response(200, json=response_payload("Fallback answer."))

    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(adapter.generate(request(timeout=TimeoutBudget(total_ms=30))))

    assert models == ["nvidia/nemotron-3.5-lightning:free", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"]
    assert result.candidate.answer_text == "Fallback answer."
    assert [attempt.outcome for attempt in result.usage.attempts] == ["timeout", "success"]
