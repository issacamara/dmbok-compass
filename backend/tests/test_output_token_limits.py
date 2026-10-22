from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

import model.gemini as gemini
import model.openrouter as openrouter
from app.contracts import CorpusServingBinding
from app.rag import LangChainRagFactory, LangChainRagSettings


class _FakeChat:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)

    def with_structured_output(self, *args: object, **kwargs: object) -> RunnableLambda:
        return RunnableLambda(lambda value: value)

    def bind(self, **kwargs: object) -> RunnableLambda:
        self.calls.append(kwargs)
        return RunnableLambda(lambda value: value)


def test_openrouter_uses_native_output_token_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChat.calls = []
    monkeypatch.setattr(openrouter, "ChatOpenRouter", _FakeChat)
    monkeypatch.setattr(openrouter.openrouter, "OpenRouter", lambda **kwargs: SimpleNamespace())

    adapter = openrouter.OpenRouterAdapter("key", max_output_tokens=700)
    adapter._build_chain("model-v1", 1000, True)

    assert _FakeChat.calls[0]["max_tokens"] == 700


def test_gemini_uses_native_output_token_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeChat.calls = []
    monkeypatch.setattr(gemini, "ChatGoogleGenerativeAI", _FakeChat)

    adapter = gemini.GeminiAdapter(max_output_tokens=700)
    adapter._build_chain("model-v1", 1.0)

    assert _FakeChat.calls[0]["max_output_tokens"] == 700


def test_missing_or_unsupported_cap_blocks_route_validation() -> None:
    factory = LangChainRagFactory.__new__(LangChainRagFactory)
    factory.settings = LangChainRagSettings(project="test")

    with pytest.raises(ValueError, match="required"):
        factory.generation_adapter()

    factory.settings = LangChainRagSettings(project="test", output_token_cap=32_769)
    with pytest.raises(ValueError, match="cap"):
        factory.generation_adapter()


def test_cap_is_bound_into_candidate_identity_artifact() -> None:
    factory = LangChainRagFactory.__new__(LangChainRagFactory)
    factory.settings = LangChainRagSettings(project="test", output_token_cap=700)
    binding = CorpusServingBinding(
        vector_collection=factory.settings.vector_collection,
        corpus_version_id="corpus-v1",
        vector_schema_version="2",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
        ingestion_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )

    first = factory.candidate_snapshot(
        binding, request_timeout_ms=10_000, generation_timeout_ms=8_000
    )
    factory.settings = LangChainRagSettings(project="test", output_token_cap=701)
    second = factory.candidate_snapshot(
        binding, request_timeout_ms=10_000, generation_timeout_ms=8_000
    )

    assert "output-token-cap=700" in first.integration_artifact
    assert first.integration_artifact != second.integration_artifact
