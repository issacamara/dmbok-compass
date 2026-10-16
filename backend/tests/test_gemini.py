import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from model.gemini import FALLBACK_MODEL, PRIMARY_MODEL, GeminiAdapter
from model.protocol import GenerationCandidate, GenerationError, GenerationErrorCode, GenerationRequest


class FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, model: str, timeout: float):
        self.calls.append({"model": model, "timeout": timeout})
        return self

    async def ainvoke(self, values, config=None) -> object:
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        try:
            parsed = GenerationCandidate.model_validate(json.loads(response.text))
        except Exception as exc:
            return {"raw": AIMessage(content=response.text), "parsed": None, "parsing_error": exc}
        usage = response.usage_metadata
        raw = AIMessage(
            content=response.text,
            usage_metadata={
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
            },
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


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

    result = asyncio.run(GeminiAdapter(chain_factory=models).generate(request()))

    assert result.candidate.answer_text == "Grounded."
    assert result.raw_output is None
    assert result.usage.model_dump() == {
        "provider": "google-gemini",
        "model": PRIMARY_MODEL,
        "input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 7,
        "attempts": [
            {"model": PRIMARY_MODEL, "outcome": "success", "status_code": 200}
        ],
    }
    assert models.calls[0]["model"] == PRIMARY_MODEL


def test_rate_limit_uses_fallback_once() -> None:
    models = FakeModels(
        [
            RuntimeError("429 resource exhausted"),
            response('{"outcome":"qualified","answer_text":"Fallback grounded."}'),
        ]
    )

    result = asyncio.run(GeminiAdapter(chain_factory=models).generate(request()))

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
        asyncio.run(GeminiAdapter(chain_factory=models).generate(request()))

    assert raised.value.code is code
    assert len(models.calls) == 1


def test_invalid_provider_json_is_not_retried() -> None:
    models = FakeModels([response("not-json")])

    with pytest.raises(GenerationError) as raised:
        asyncio.run(GeminiAdapter(chain_factory=models).generate(request()))

    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT
    assert len(models.calls) == 1


def test_provider_wide_failure_does_not_use_same_provider_fallback() -> None:
    provider_failure = RuntimeError("provider outage")
    provider_failure.provider_wide = True
    models = FakeModels([provider_failure])

    with pytest.raises(GenerationError) as raised:
        asyncio.run(GeminiAdapter(chain_factory=models).generate(request()))

    assert raised.value.code is GenerationErrorCode.UNKNOWN
    assert len(models.calls) == 1
