"""Grounded, request-scoped answer policy.

This module deliberately has no persistence boundary.  Callers provide the
retrieved passages and quota snapshot, and the returned response contains the
only copy of the question-derived data.  The model receives a bounded,
explicit evidence set and its citations are replaced with immutable passage
provenance before the response leaves the service.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Protocol

from langchain_core.runnables import Runnable, RunnableConfig

from app.contracts import (
    AnswerResponse,
    ApiError,
    CorpusVersion,
    QuotaStatus,
    RetrievedPassage,
    RetrievalTrace,
)
from app.retrieval.search import retrieve_off_event_loop
from model.langchain import (
    LangChainGenerationRunnable,
    as_langchain_runnable,
    reject_caller_runnable_configuration,
)
from model.protocol import (
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    ModelAdapter,
    Passage,
    TimeoutBudget,
)

from .citations import validated_citations

EvidenceOutcome = Literal["strong", "partial", "absent"]
STRONG_EVIDENCE_THRESHOLD = 0.75
PARTIAL_EVIDENCE_THRESHOLD = 0.45
MAX_EVIDENCE_PASSAGES = 5
DEFAULT_REQUEST_TIMEOUT_MS = 14_000
DEFAULT_GENERATION_TIMEOUT_MS = 8_000

REFUSAL_TEXT = "I could not find enough relevant evidence in the DMBOK corpus to answer that question."
PROMPT_VERSION = "grounded-answer-policy-v2"
GROUNDING_PROMPT = """<user_question>
{question}
</user_question>

<evidence_strength>{evidence_outcome}</evidence_strength>"""


@dataclass(frozen=True)
class EvidenceBundle:
    outcome: EvidenceOutcome
    passages: tuple[RetrievedPassage, ...]
    basis: str


@dataclass(frozen=True)
class RequestDeadline:
    """One monotonic deadline shared by retrieval and model generation."""

    started: float
    total_ms: int
    clock: Callable[[], float] = perf_counter

    @classmethod
    def start(
        cls, total_ms: int, *, clock: Callable[[], float] = perf_counter
    ) -> "RequestDeadline":
        if total_ms < 1:
            raise ValueError("request timeout must be positive")
        return cls(started=clock(), total_ms=total_ms, clock=clock)

    def elapsed_ms(self) -> float:
        return round(max(0.0, (self.clock() - self.started) * 1000), 3)

    def remaining_ms(self) -> int:
        return max(0, self.total_ms - int(self.elapsed_ms()))

    def remaining_seconds(self) -> float:
        return self.remaining_ms() / 1000


def timeout_settings(request_timeout: str | int, generation_timeout: str | int) -> tuple[int, int]:
    """Parse the externally configured whole-path and generation budgets."""

    try:
        request_ms = int(request_timeout)
        generation_ms = int(generation_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("answer timeout settings must be integer milliseconds") from exc
    if not 1 <= generation_ms <= request_ms <= 60_000:
        raise ValueError(
            "answer generation timeout must be positive, no greater than the request timeout, "
            "and the request timeout must not exceed 60000 ms"
        )
    return request_ms, generation_ms


def classify_evidence(
    passages: Iterable[RetrievedPassage],
    *,
    strong_threshold: float = STRONG_EVIDENCE_THRESHOLD,
    partial_threshold: float = PARTIAL_EVIDENCE_THRESHOLD,
) -> EvidenceBundle:
    """Keep finite, normalized evidence and classify it deterministically."""

    if not (
        isfinite(strong_threshold)
        and isfinite(partial_threshold)
        and 0 <= partial_threshold < strong_threshold <= 1
    ):
        raise ValueError("evidence thresholds must satisfy 0 <= partial < strong <= 1")

    candidates = sorted(
        (
            passage
            for passage in passages
            if passage.relevance_score is not None
            and isfinite(passage.relevance_score)
            and partial_threshold <= passage.relevance_score <= 1
        ),
        key=lambda passage: (-passage.relevance_score, passage.chunk_id),
    )[:MAX_EVIDENCE_PASSAGES]
    top_score = candidates[0].relevance_score if candidates else None
    outcome: EvidenceOutcome
    if top_score is None:
        outcome = "absent"
    elif top_score >= strong_threshold:
        outcome = "strong"
    else:
        outcome = "partial"
    score = "none" if top_score is None else f"{top_score:.4f}"
    return EvidenceBundle(
        outcome=outcome,
        passages=tuple(candidates),
        basis=(
            f"top_score={score}; strong_threshold={strong_threshold:.4f}; "
            f"partial_threshold={partial_threshold:.4f}"
        ),
    )


class GroundedAnswerPolicy:
    """Turn retrieved evidence and one model candidate into an API response."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        model_name: str = "answer-model",
        timeout_ms: int = DEFAULT_GENERATION_TIMEOUT_MS,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        # The provider boundary is always invoked through the constrained
        # runnable.  This prevents LangChain callbacks from receiving prompts,
        # evidence, or generated answers.
        self._generator: LangChainGenerationRunnable = as_langchain_runnable(adapter)
        self._model_name = model_name
        self._timeout_ms = timeout_ms

    async def answer(
        self,
        question: str,
        passages: Iterable[RetrievedPassage],
        quota: QuotaStatus,
        *,
        retrieval_ms: float | None = None,
        deadline: RequestDeadline | None = None,
    ) -> AnswerResponse:
        """Return a cited answer, qualified answer, or evidence-based refusal."""

        if not question or not question.strip():
            raise ValueError("question must be non-empty")

        deadline = deadline or RequestDeadline.start(self._timeout_ms)
        retrieved = tuple(passages)[:MAX_EVIDENCE_PASSAGES]
        evidence = classify_evidence(retrieved)
        timings = {"retrieval_ms": retrieval_ms} if retrieval_ms is not None else {}
        trace = RetrievalTrace(
            retrieved_passages=list(retrieved),
            selected_model=self._model_name,
            timings_ms=timings,
        )
        if evidence.outcome == "absent":
            return self._finalize(
                AnswerResponse(
                    outcome="refusal",
                    answer_text=REFUSAL_TEXT,
                    trace=trace,
                    quota=quota,
                ),
                deadline,
            )

        request = GenerationRequest(
            prompt=GROUNDING_PROMPT.format(
                question=question.strip(), evidence_outcome=evidence.outcome
            ),
            passages=[
                Passage(
                    passage_id=passage.chunk_id,
                    page=passage.page,
                    section=passage.section,
                    text=passage.excerpt,
                )
                for passage in evidence.passages
            ],
        )
        generation_started = deadline.clock()
        try:
            generation_budget_ms = min(self._timeout_ms, deadline.remaining_ms())
            if generation_budget_ms < 1:
                raise GenerationError(
                    GenerationErrorCode.TIMEOUT,
                    "Answer generation exceeded the whole-request deadline.",
                    retryable=True,
                )
            result = await self._generator.ainvoke(
                request.model_copy(update={"timeout": TimeoutBudget(total_ms=generation_budget_ms)})
            )
        except GenerationError as exc:
            if exc.attempts:
                trace = trace.model_copy(update={
                    "selected_model": exc.attempts[-1].model,
                    "model_attempts": list(exc.attempts),
                })
            return self._finalize(
                self._operational_refusal(trace, quota, exc),
                deadline,
                generation_started,
            )

        trace = trace.model_copy(
            update={
                "selected_model": result.usage.model,
                "model_attempts": result.usage.attempts,
                "timings_ms": {
                    **trace.timings_ms,
                    "generation_ms": _elapsed_ms(generation_started),
                },
                "model_output": result.raw_output,
            }
        )
        if deadline.remaining_ms() < 1:
            return self._finalize(
                self._operational_refusal(
                    trace,
                    quota,
                    GenerationError(
                        GenerationErrorCode.TIMEOUT,
                        "Answer generation exceeded the whole-request deadline.",
                        retryable=True,
                        attempts=tuple(result.usage.attempts),
                    ),
                ),
                deadline,
                generation_started,
            )
        candidate = result.candidate
        if candidate.outcome == "refusal":
            return self._finalize(
                AnswerResponse(
                    outcome="refusal",
                    answer_text=candidate.answer_text or REFUSAL_TEXT,
                    trace=trace,
                    quota=quota,
                ),
                deadline,
            )

        citations = validated_citations(candidate, evidence.passages)
        if not citations:
            return self._finalize(
                self._policy_refusal(
                    trace,
                    quota,
                    "The model response did not contain a citation to supplied evidence.",
                ),
                deadline,
            )

        outcome = "qualified" if evidence.outcome == "partial" else candidate.outcome
        return self._finalize(
            AnswerResponse(
                outcome=outcome,
                answer_text=candidate.answer_text,
                synthesis=candidate.synthesis or len(citations) > 1,
                citations=citations,
                trace=trace,
                quota=quota,
            ),
            deadline,
        )

    @staticmethod
    def _finalize(
        response: AnswerResponse,
        deadline: RequestDeadline,
        generation_started: float | None = None,
    ) -> AnswerResponse:
        timings = dict(response.trace.timings_ms)
        if generation_started is not None and "generation_ms" not in timings:
            timings["generation_ms"] = round(
                max(0.0, (deadline.clock() - generation_started) * 1000), 3
            )
        timings["total_ms"] = deadline.elapsed_ms()
        return response.model_copy(
            update={"trace": response.trace.model_copy(update={"timings_ms": timings})}
        )

    def service_refusal(
        self,
        quota: QuotaStatus,
        *,
        code: str,
        message: str,
        deadline: RequestDeadline | None = None,
        retrieval_ms: float | None = None,
    ) -> AnswerResponse:
        deadline = deadline or RequestDeadline.start(self._timeout_ms)
        timings = {"retrieval_ms": retrieval_ms} if retrieval_ms is not None else {}
        return self._finalize(AnswerResponse(
            outcome="refusal",
            error=ApiError(code=code, message=message, retryable=True, status=503),
            trace=RetrievalTrace(
                retrieved_passages=[], selected_model=self._model_name, timings_ms=timings
            ),
            quota=quota,
        ), deadline)

    def _operational_refusal(
        self, trace: RetrievalTrace, quota: QuotaStatus, error: GenerationError
    ) -> AnswerResponse:
        return AnswerResponse(
            outcome="refusal",
            error=ApiError(
                code=f"answer_provider_{error.code.value}",
                message=error.message,
                retryable=error.retryable,
                status=503 if error.retryable else 502,
            ),
            trace=trace,
            quota=quota,
        )

    @staticmethod
    def _policy_refusal(
        trace: RetrievalTrace, quota: QuotaStatus, message: str
    ) -> AnswerResponse:
        return AnswerResponse(
            outcome="refusal",
            error=ApiError(
                code="grounding_policy_rejected",
                message=message,
                status=502,
            ),
            trace=trace,
            quota=quota,
        )


def _elapsed_ms(started: float) -> float:
    return round(max(0.0, (perf_counter() - started) * 1000), 3)


class QuestionAnswerer(Protocol):
    async def answer(self, question: str, quota: QuotaStatus) -> AnswerResponse:
        """Answer one approved-user question without persisting its content."""


class PassageRetriever(Protocol):
    def retrieve(
        self, question: str, *, active_corpus_version_id: str, limit: int = MAX_EVIDENCE_PASSAGES
    ) -> tuple[RetrievedPassage, ...]:
        """Retrieve ranked passages from one active corpus version."""


class ActiveCorpusStore(Protocol):
    def active(self) -> Any | None:
        """Return the active corpus version, if one exists."""


@dataclass(frozen=True)
class AnswerInvocation:
    """Bounded, ephemeral input to the LangChain RAG composition boundary."""

    question: str
    quota: QuotaStatus


class LangChainRagPipeline(Runnable[AnswerInvocation, AnswerResponse]):
    """Compose active-corpus retrieval and grounded answering as one runnable.

    The pipeline intentionally owns no authorization, quota reservation, or
    persistence.  Its input and output are request-scoped application types;
    callbacks are rejected because they could retain content outside that
    boundary.
    """

    def __init__(
        self,
        retriever: PassageRetriever,
        corpus_versions: ActiveCorpusStore,
        policy: GroundedAnswerPolicy,
        *,
        request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
        clock: Callable[[], float] = perf_counter,
        retriever_for_binding: Callable[[CorpusVersion], PassageRetriever] | None = None,
    ) -> None:
        self._retriever = retriever
        self._corpus_versions = corpus_versions
        self._policy = policy
        self._request_timeout_ms = request_timeout_ms
        self._clock = clock
        self._retriever_for_binding = retriever_for_binding

    def invoke(
        self,
        input: AnswerInvocation,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AnswerResponse:
        reject_caller_runnable_configuration(config, kwargs)
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, config, **kwargs))
        raise RuntimeError("Use ainvoke() when already running an event loop.")

    async def ainvoke(
        self,
        input: AnswerInvocation,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AnswerResponse:
        reject_caller_runnable_configuration(config, kwargs)
        if not isinstance(input, AnswerInvocation):
            raise TypeError("input must be an AnswerInvocation")
        deadline = RequestDeadline.start(self._request_timeout_ms, clock=self._clock)
        active = self._corpus_versions.active()
        if active is None:
            return self._policy.service_refusal(
                input.quota,
                code="active_corpus_unavailable",
                message="No active DMBOK corpus is available.",
                deadline=deadline,
            )
        try:
            retriever = (
                self._retriever_for_binding(active)
                if self._retriever_for_binding is not None
                else self._retriever
            )
            async with asyncio.timeout(deadline.remaining_seconds()):
                passages, retrieval_ms = await retrieve_off_event_loop(
                    retriever.retrieve,
                    input.question,
                    active_corpus_version_id=active.version_id,
                    limit=MAX_EVIDENCE_PASSAGES,
                )
        except TimeoutError:
            return self._policy.service_refusal(
                input.quota,
                code="answer_request_timeout",
                message="Answer retrieval exceeded the whole-request deadline.",
                deadline=deadline,
            )
        return await self._policy.answer(
            input.question,
            passages,
            input.quota,
            retrieval_ms=round(max(0.0, retrieval_ms), 3),
            deadline=deadline,
        )

class RetrievedAnswerService:
    """Bind active-corpus retrieval to the pure grounded answer policy."""

    def __init__(
        self,
        retriever: PassageRetriever,
        corpus_versions: ActiveCorpusStore,
        policy: GroundedAnswerPolicy,
        *,
        request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
        retriever_for_binding: Callable[[CorpusVersion], PassageRetriever] | None = None,
    ) -> None:
        self._pipeline = LangChainRagPipeline(
            retriever,
            corpus_versions,
            policy,
            request_timeout_ms=request_timeout_ms,
            retriever_for_binding=retriever_for_binding,
        )

    async def answer(self, question: str, quota: QuotaStatus) -> AnswerResponse:
        return await self._pipeline.ainvoke(AnswerInvocation(question=question, quota=quota))

    def as_langchain_runnable(self) -> LangChainRagPipeline:
        """Expose the complete ephemeral RAG path for bounded composition."""
        return self._pipeline
