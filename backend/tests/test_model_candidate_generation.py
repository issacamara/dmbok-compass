from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from model.candidate_generation import (
    CandidateDraftBatch,
    EvaluationCandidateAdapter,
)
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest, Passage


def draft_batch():
    return CandidateDraftBatch.model_validate({
        "candidates": [{
            "question": "What does the corpus say about quality ownership?",
            "category": "definitions",
            "expected_outcome": "answer",
            "expected_answer_or_rubric": "Explain the DMBOK concept.",
            "supports": [{"source_id": "source-0", "quote": "Quality has an owner."}],
            "scope_rationale": None,
        }],
    })


class FakeChain:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.values = None

    async def ainvoke(self, values, *, config):
        self.values = values
        assert config == {"callbacks": []}
        if self.error:
            raise self.error
        return self.response


def test_candidate_adapter_uses_pinned_generator_route_and_returns_content_free_usage():
    chain = FakeChain({
        "parsed": draft_batch(),
        "raw": SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}),
        "parsing_error": None,
    })
    adapter = EvaluationCandidateAdapter(
        route="openrouter",
        model="generator-v1",
        api_key="test-secret",
        timeout_ms=5000,
        max_output_tokens=2048,
        chain_factory=lambda model, timeout: chain,
    )
    request = GenerationRequest(
        prompt="Generate a supported evaluation candidate.",
        passages=[Passage(passage_id="source-0", page=1, section="Quality", text="Quality has an owner.")],
    )

    result = asyncio.run(adapter.generate(request))

    assert result.candidates == draft_batch().candidates
    assert result.usage.provider == "openrouter"
    assert result.usage.model == "generator-v1"
    assert result.usage.total_tokens == 18
    assert "source-0" in chain.values["evidence"]
    assert adapter.route == "openrouter"
    assert adapter.model == "generator-v1"


def test_candidate_adapter_rejects_invalid_structured_output():
    adapter = EvaluationCandidateAdapter(
        route="vertex",
        model="generator-v1",
        project="project",
        location="europe-west1",
        timeout_ms=5000,
        max_output_tokens=2048,
        chain_factory=lambda model, timeout: FakeChain({"parsed": None, "raw": object()}),
    )

    with pytest.raises(GenerationError) as raised:
        asyncio.run(adapter.generate(GenerationRequest(prompt="Generate candidates.")))

    assert raised.value.code == GenerationErrorCode.INVALID_OUTPUT
    assert raised.value.retryable is False


def test_candidate_adapter_maps_timeouts_to_retryable_safe_errors():
    adapter = EvaluationCandidateAdapter(
        route="openrouter",
        model="generator-v1",
        api_key="test-secret",
        timeout_ms=5000,
        max_output_tokens=2048,
        chain_factory=lambda model, timeout: FakeChain(error=TimeoutError()),
    )

    with pytest.raises(GenerationError) as raised:
        asyncio.run(adapter.generate(GenerationRequest(prompt="Generate candidates.")))

    assert raised.value.code == GenerationErrorCode.TIMEOUT
    assert raised.value.retryable is True
