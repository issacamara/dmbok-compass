"""Grounded, request-scoped answer policy.

This module deliberately has no persistence boundary.  Callers provide the
retrieved passages and quota snapshot, and the returned response contains the
only copy of the question-derived data.  The model receives a bounded,
explicit evidence set and its citations are replaced with immutable passage
provenance before the response leaves the service.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Iterable, Literal, Protocol

from app.contracts import (
    AnswerResponse,
    ApiError,
    QuotaStatus,
    RetrievedPassage,
    RetrievalTrace,
)
from model.protocol import (
    GenerationError,
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

REFUSAL_TEXT = "I could not find enough relevant evidence in the DMBOK corpus to answer that question."
GROUNDING_PROMPT = """You are a DMBOK Compass answerer.
Answer the user's question using only the supplied DMBOK evidence.
Do not use general model knowledge, invent facts, or cite anything outside
the supplied passages. If the evidence is incomplete, state the limitation
and return a qualified answer. Mark synthesis when combining passages.
Every material claim in an answer must be supported by one or more supplied
passage IDs. Return structured JSON matching the response schema.

User question:
{question}

Evidence strength:
{evidence_outcome}
"""


@dataclass(frozen=True)
class EvidenceBundle:
    outcome: EvidenceOutcome
    passages: tuple[RetrievedPassage, ...]
    basis: str


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
        timeout_ms: int = 8_000,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        self._adapter = adapter
        self._model_name = model_name
        self._timeout_ms = timeout_ms

    async def answer(
        self,
        question: str,
        passages: Iterable[RetrievedPassage],
        quota: QuotaStatus,
        *,
        retrieval_ms: float | None = None,
    ) -> AnswerResponse:
        """Return a cited answer, qualified answer, or evidence-based refusal."""

        if not question or not question.strip():
            raise ValueError("question must be non-empty")

        started = perf_counter()
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
                started,
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
        generation_started = perf_counter()
        try:
            result = await self._adapter.generate(
                request.model_copy(update={"timeout": TimeoutBudget(total_ms=self._timeout_ms)})
            )
        except GenerationError as exc:
            return self._finalize(
                self._operational_refusal(trace, quota, exc),
                started,
                generation_started,
            )

        trace = trace.model_copy(
            update={
                "selected_model": result.usage.model,
                "timings_ms": {
                    **trace.timings_ms,
                    "generation_ms": _elapsed_ms(generation_started),
                },
            }
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
                started,
            )

        citations = validated_citations(candidate, evidence.passages)
        if not citations:
            return self._finalize(
                self._policy_refusal(
                    trace,
                    quota,
                    "The model response did not contain a citation to supplied evidence.",
                ),
                started,
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
            started,
        )

    @staticmethod
    def _finalize(
        response: AnswerResponse,
        started: float,
        generation_started: float | None = None,
    ) -> AnswerResponse:
        timings = dict(response.trace.timings_ms)
        if generation_started is not None and "generation_ms" not in timings:
            timings["generation_ms"] = _elapsed_ms(generation_started)
        timings["total_ms"] = _elapsed_ms(started)
        return response.model_copy(
            update={"trace": response.trace.model_copy(update={"timings_ms": timings})}
        )

    def service_refusal(
        self, quota: QuotaStatus, *, code: str, message: str
    ) -> AnswerResponse:
        return AnswerResponse(
            outcome="refusal",
            error=ApiError(code=code, message=message, retryable=True, status=503),
            trace=RetrievalTrace(retrieved_passages=[], selected_model=self._model_name),
            quota=quota,
        )

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


class RetrievedAnswerService:
    """Bind active-corpus retrieval to the pure grounded answer policy."""

    def __init__(
        self,
        retriever: PassageRetriever,
        corpus_versions: ActiveCorpusStore,
        policy: GroundedAnswerPolicy,
    ) -> None:
        self._retriever = retriever
        self._corpus_versions = corpus_versions
        self._policy = policy

    async def answer(self, question: str, quota: QuotaStatus) -> AnswerResponse:
        active = self._corpus_versions.active()
        if active is None:
            return self._policy.service_refusal(
                quota,
                code="active_corpus_unavailable",
                message="No active DMBOK corpus is available.",
            )
        retrieval_started = perf_counter()
        passages = self._retriever.retrieve(
            question,
            active_corpus_version_id=active.version_id,
            limit=MAX_EVIDENCE_PASSAGES,
        )
        return await self._policy.answer(
            question,
            passages,
            quota,
            retrieval_ms=_elapsed_ms(retrieval_started),
        )
