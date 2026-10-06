import asyncio
from types import SimpleNamespace

import pytest

from model.gemini import FALLBACK_MODEL, PRIMARY_MODEL, GeminiAdapter
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest


class FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.aio = SimpleNamespace(models=models)


def request() -> GenerationRequest:
    return GenerationRequest(prompt="Answer only from evidence.")


def response(text: str, model_tokens: tuple[int, int, int] = (4, 3, 7)) -> object:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=model_tokens[0],
            candidates_token_count=model_tokens[1],
            total_token_count=model_tokens[2],
        ),
    )


def test_gemini_adapter_maps_structured_output_and_usage() -> None:
    models = FakeModels([response('{"outcome":"answer","answer_text":"Grounded."}')])

    result = asyncio.run(GeminiAdapter(FakeClient(models)).generate(request()))

    assert result.candidate.answer_text == "Grounded."
    assert result.usage.model_dump() == {
        "provider": "google-gemini",
        "model": PRIMARY_MODEL,
        "input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 7,
    }
    assert models.calls[0]["model"] == PRIMARY_MODEL
    assert models.calls[0]["config"]["response_mime_type"] == "application/json"


def test_rate_limit_uses_fallback_once() -> None:
    models = FakeModels(
        [
            RuntimeError("429 resource exhausted"),
            response('{"outcome":"qualified","answer_text":"Fallback grounded."}'),
        ]
    )

    result = asyncio.run(GeminiAdapter(FakeClient(models)).generate(request()))

    assert result.candidate.answer_text == "Fallback grounded."
    assert [call["model"] for call in models.calls] == [PRIMARY_MODEL, FALLBACK_MODEL]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("401 unauthenticated"), GenerationErrorCode.AUTHENTICATION_FAILED),
        (RuntimeError("400 invalid argument"), GenerationErrorCode.REQUEST_REJECTED),
    ],
)
def test_non_retryable_provider_failures_do_not_use_fallback(
    error: Exception, code: GenerationErrorCode
) -> None:
    models = FakeModels([error])

    with pytest.raises(GenerationError) as raised:
        asyncio.run(GeminiAdapter(FakeClient(models)).generate(request()))

    assert raised.value.code is code
    assert len(models.calls) == 1


def test_invalid_provider_json_is_not_retried() -> None:
    models = FakeModels([response("not-json")])

    with pytest.raises(GenerationError) as raised:
        asyncio.run(GeminiAdapter(FakeClient(models)).generate(request()))

    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT
    assert len(models.calls) == 1


def test_provider_wide_failure_does_not_use_same_provider_fallback() -> None:
    provider_failure = RuntimeError("provider outage")
    provider_failure.provider_wide = True
    models = FakeModels([provider_failure])

    with pytest.raises(GenerationError) as raised:
        asyncio.run(GeminiAdapter(FakeClient(models)).generate(request()))

    assert raised.value.code is GenerationErrorCode.UNKNOWN
    assert len(models.calls) == 1
