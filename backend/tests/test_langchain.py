import asyncio
import json
import threading
from datetime import datetime, timezone

import httpx
import pytest

from app.answering import AnswerInvocation, GroundedAnswerPolicy, RetrievedAnswerService
from app.contracts import CorpusVersion, QuotaStatus, RetrievedPassage
from model.fake import FakeAdapter
from model.openrouter import OpenRouterAdapter
from model.protocol import GenerationError, GenerationErrorCode, GenerationRequest, TimeoutBudget


def request(**kwargs: object) -> GenerationRequest:
    return GenerationRequest(prompt="Answer only from supplied evidence.", **kwargs)


def response(answer: str = "Grounded answer.") -> dict[str, object]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps({
            "outcome": "answer", "answer_text": answer, "synthesis": False, "citations": []
        })}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "id": "test-completion", "created": 1, "model": "test-model",
        "object": "chat.completion", "system_fingerprint": None,
    }


def test_runnable_preserves_primary_result_and_safe_usage() -> None:
    adapter = OpenRouterAdapter("test-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response())))
    result = asyncio.run(adapter.as_langchain_runnable().ainvoke(request()))

    assert result.candidate.answer_text == "Grounded answer."
    assert result.usage.model == "mistralai/mistral-nemo"
    assert result.usage.model_dump(exclude={"attempts"}) == {
        "provider": "openrouter", "model": "mistralai/mistral-nemo",
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
    payload = response()
    payload["choices"][0]["message"]["content"] = "not JSON"
    with pytest.raises(GenerationError) as raised:
        asyncio.run(OpenRouterAdapter("test-key", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)
        )).as_langchain_runnable().ainvoke(request()))
    assert raised.value.code is GenerationErrorCode.INVALID_OUTPUT


def test_runnable_rejects_content_bearing_callbacks() -> None:
    runnable = OpenRouterAdapter("test-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response()))).as_langchain_runnable()
    with pytest.raises(ValueError, match="callbacks"):
        asyncio.run(runnable.ainvoke(request(), {"callbacks": [object()]}))


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
    )


def passage(chunk_id: str = "chunk-1") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=1,
        section="Governance",
        excerpt="Stored DMBOK evidence.",
        relevance_score=0.9,
    )


class FakeCorpusStore:
    def __init__(self, version_id: str) -> None:
        self._active = CorpusVersion(
            version_id=version_id,
            source_uri="gs://bucket/dmbok.pdf",
            embedding_model="gemini-embedding-001",
            status="active",
            created_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
        )

    def active(self) -> CorpusVersion:
        return self._active


class BlockingRetriever:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, question: str, *, active_corpus_version_id: str, limit: int):
        self.calls.append((question, active_corpus_version_id, limit))
        self.started.set()
        assert self.release.wait(timeout=1)
        return (passage("shared"),)


class RecordingRetriever:
    def __init__(self, expected_calls: int) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self._barrier = threading.Barrier(expected_calls)

    def retrieve(self, question: str, *, active_corpus_version_id: str, limit: int):
        self.calls.append((question, active_corpus_version_id, limit))
        self._barrier.wait(timeout=1)
        return (passage("shared"),)


class CountingAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__(
            {
                "outcome": "answer",
                "answer_text": "Grounded.",
                "citations": [{"citation_id": "chunk-1"}],
            }
        )
        self.calls = 0

    async def generate(self, request: GenerationRequest):
        self.calls += 1
        return await super().generate(request)


def service(retriever, version_id: str = "v1", adapter: FakeAdapter | None = None):
    policy = GroundedAnswerPolicy(
        adapter
        or FakeAdapter(
            {
                "outcome": "answer",
                "answer_text": "Grounded.",
                "citations": [
                    {
                        "citation_id": "shared",
                        "page": 1,
                        "section": "Governance",
                        "excerpt": "Stored DMBOK evidence.",
                    }
                ],
            }
        )
    )
    return RetrievedAnswerService(retriever, FakeCorpusStore(version_id), policy)


def test_blocking_retrieval_does_not_block_other_requests_or_event_loop() -> None:
    async def exercise() -> None:
        blocked = BlockingRetriever()
        slow = service(blocked)
        fast = service(RecordingRetriever(1), version_id="v2")
        slow_task = asyncio.create_task(
            slow.answer("slow question", quota())
        )
        assert await asyncio.to_thread(blocked.started.wait, 1)

        heartbeat = asyncio.Event()

        async def tick() -> None:
            await asyncio.sleep(0)
            heartbeat.set()

        fast_response, _ = await asyncio.gather(
            fast.answer("fast question", quota()), tick()
        )
        assert heartbeat.is_set()
        assert fast_response.outcome == "answer"

        blocked.release.set()
        assert (await slow_task).outcome == "answer"

    asyncio.run(exercise())


def test_concurrent_invocations_keep_corpus_bindings_request_local() -> None:
    async def exercise() -> None:
        retriever = RecordingRetriever(3)
        responses = await asyncio.gather(
            *(
                service(retriever, version_id).as_langchain_runnable().ainvoke(
                    AnswerInvocation(question=question, quota=quota())
                )
                for question, version_id in (
                    ("first", "v1"),
                    ("second", "v2"),
                    ("third", "v3"),
                )
            )
        )
        assert [response.outcome for response in responses] == ["answer"] * 3
        assert set(retriever.calls) == {
            ("first", "v1", 5),
            ("second", "v2", 5),
            ("third", "v3", 5),
        }

    asyncio.run(exercise())


def test_cancelling_retrieval_does_not_start_generation_or_extra_retrieval_work() -> None:
    async def exercise() -> None:
        retriever = BlockingRetriever()
        adapter = CountingAdapter()
        pending = asyncio.create_task(service(retriever, adapter=adapter).answer("question", quota()))
        assert await asyncio.to_thread(retriever.started.wait, 1)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        retriever.release.set()
        await asyncio.sleep(0)
        assert retriever.calls == [("question", "v1", 5)]
        assert adapter.calls == 0

    asyncio.run(exercise())
