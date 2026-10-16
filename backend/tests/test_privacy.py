import asyncio
import logging
import os
from types import SimpleNamespace

import pytest
from langchain_core.globals import get_debug, get_verbose, set_debug, set_verbose
from langchain_core.runnables import RunnableLambda

from app.contracts import AnswerResponse, QuotaStatus, RetrievalTrace
from app.telemetry import InMemoryTelemetryAdapter, LoggingTelemetryAdapter, error_event, response_event
from model import gemini, openrouter
from model.gemini import GeminiAdapter
from model.openrouter import OpenRouterAdapter
from model.protocol import GenerationCandidate, GenerationError, GenerationRequest, Passage


SENTINELS = {
    "question": "SENTINEL_QUESTION",
    "prompt": "SENTINEL_PROMPT",
    "passage": "SENTINEL_PASSAGE",
    "answer": "SENTINEL_ANSWER",
    "raw_output": "SENTINEL_RAW_OUTPUT",
    "credential": "SENTINEL_CREDENTIAL",
}
PRIVATE_ENVIRONMENT = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_DEBUG",
    "LANGCHAIN_VERBOSE",
)


def request() -> GenerationRequest:
    return GenerationRequest(
        prompt=SENTINELS["prompt"],
        passages=[
            Passage(
                passage_id="sentinel-passage",
                page=1,
                section="Sentinel",
                text=SENTINELS["passage"],
            )
        ],
    )


class RecordingChain:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[dict[str, str], object]] = []

    async def ainvoke(self, values: dict[str, str], config: object) -> object:
        self.calls.append((values, config))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def successful_native_response() -> dict[str, object]:
    candidate = GenerationCandidate(
        outcome="answer", answer_text=SENTINELS["answer"], citations=[]
    )
    return {
        "raw": SimpleNamespace(
            content=(
                '{"outcome":"answer","answer_text":"'
                + SENTINELS["raw_output"]
                + '"}'
            ),
            usage_metadata={},
        ),
        "parsed": candidate,
        "parsing_error": None,
    }


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at="2030-01-01T00:00:00Z",
    )


def test_native_call_disables_environment_tracing_debug_verbose_and_callbacks(monkeypatch) -> None:
    for name in PRIVATE_ENVIRONMENT:
        monkeypatch.setenv(name, "true")
    set_debug(True)
    set_verbose(True)
    chain = RecordingChain(successful_native_response())
    adapter = OpenRouterAdapter(
        SENTINELS["credential"], fallback_model=None, chain_factory=lambda *_: chain
    )

    result = asyncio.run(adapter.generate(request()))

    assert result.candidate.answer_text == SENTINELS["answer"]
    assert result.raw_output is None
    assert all(os.environ[name] == "false" for name in PRIVATE_ENVIRONMENT)
    assert get_debug() is False
    assert get_verbose() is False
    values, config = chain.calls[0]
    assert values["prompt"] == SENTINELS["prompt"]
    assert SENTINELS["passage"] in values["evidence"]
    assert config == {"callbacks": []}


def test_success_and_failure_sentinels_never_reach_telemetry_or_in_memory_stores(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="dmbok_compass")
    success = asyncio.run(
        OpenRouterAdapter(
            SENTINELS["credential"],
            fallback_model=None,
            chain_factory=lambda *_: RecordingChain(successful_native_response()),
        ).generate(request())
    )
    response = AnswerResponse(
        outcome="answer",
        answer_text=success.candidate.answer_text,
        trace=RetrievalTrace(
            retrieved_passages=[],
            selected_model=success.usage.model,
            model_output=success.raw_output,
        ),
        quota=quota(),
    )
    persistent_test_adapter = InMemoryTelemetryAdapter()
    LoggingTelemetryAdapter().emit(response_event(response, 0))
    persistent_test_adapter.emit(response_event(response, 0))

    failure = RuntimeError(" ".join(SENTINELS.values()))
    with pytest.raises(GenerationError) as raised:
        asyncio.run(
            OpenRouterAdapter(
                SENTINELS["credential"],
                fallback_model=None,
                chain_factory=lambda *_: RecordingChain(failure),
            ).generate(request())
        )
    LoggingTelemetryAdapter().emit(error_event(raised.value))
    persistent_test_adapter.emit(error_event(raised.value))

    durable_values = str([event.model_dump() for event in persistent_test_adapter.events])
    rendered_logs = " ".join(record.getMessage() for record in caplog.records)
    for sentinel in SENTINELS.values():
        assert sentinel not in durable_values
        assert sentinel not in rendered_logs


@pytest.mark.parametrize("module, adapter", [(openrouter, "openrouter"), (gemini, "gemini")])
def test_native_chat_models_disable_caches_without_history_or_checkpoints(
    monkeypatch, module, adapter: str
) -> None:
    captured: dict[str, object] = {}

    class FakeChat:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def with_structured_output(self, *args: object, **kwargs: object) -> RunnableLambda:
            return RunnableLambda(lambda value: value)

        def bind(self, **kwargs: object) -> RunnableLambda:
            return RunnableLambda(lambda value: value)

    if adapter == "openrouter":
        monkeypatch.setattr(module, "ChatOpenRouter", FakeChat)
        monkeypatch.setattr(module.openrouter, "OpenRouter", lambda **_: object())
        OpenRouterAdapter("test-key")._build_chain("test-model", 1000, True)
    else:
        monkeypatch.setattr(module, "ChatGoogleGenerativeAI", FakeChat)
        GeminiAdapter()._build_chain("test-model", 1.0)

    assert captured["cache"] is False
    assert "history" not in captured
    assert "checkpointer" not in captured
