"""Opt-in hybrid retrieval experiments.

The dense retriever remains the serving baseline.  This module provides the
small, deterministic experiment boundary used by evaluation: it combines a
candidate-filtered dense result with an immutable lexical result, optionally
reranks that bounded union, and returns the frozen :class:`RetrievalResult`
contract.  It deliberately has no persistence or serving configuration
side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable, Protocol

from app.contracts import CandidateIdentity, RetrievedPassage, RetrievalResult
from app.retrieval.lexical import LexicalCandidate, LexicalIndex

MAX_OUTPUT_CANDIDATES = 5
DEFAULT_CANDIDATE_LIMIT = 10
DEFAULT_RRF_K = 60


class HybridRetrievalError(ValueError):
    """Raised when the selected experiment cannot produce a safe result."""


class BoundedReranker(Protocol):
    def rerank(
        self, question: str, candidates: tuple[RetrievedPassage, ...]
    ) -> Iterable[RetrievedPassage]: ...


@dataclass(frozen=True, slots=True)
class HybridStrategy:
    """Versioned settings for one evaluated, non-default strategy."""

    strategy_id: str = "hybrid-rrf-v1"
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    output_limit: int = MAX_OUTPUT_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    reranker: BoundedReranker | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must be non-empty")
        if not 1 <= self.output_limit <= MAX_OUTPUT_CANDIDATES:
            raise ValueError("output_limit must be between 1 and 5")
        if self.candidate_limit < self.output_limit:
            raise ValueError("candidate_limit must be at least output_limit")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")


DenseRetriever = Callable[..., tuple[RetrievedPassage, ...]]


class HybridRetriever:
    """Run one explicitly selected hybrid experiment against a pinned corpus."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_index: LexicalIndex,
        candidate: CandidateIdentity,
        *,
        strategy: HybridStrategy = HybridStrategy(),
    ) -> None:
        self._dense_retriever = dense_retriever
        self._lexical_index = lexical_index
        self._candidate = candidate
        self._strategy = strategy

    def retrieve(self, question: str) -> RetrievalResult:
        if not question or not question.strip():
            raise HybridRetrievalError("question must be non-empty")
        dense = self._dense_retriever(
            question,
            active_corpus_version_id=self._candidate.corpus_version_id,
            limit=self._strategy.candidate_limit,
        )
        lexical = self._lexical_index.search(
            question, limit=min(self._strategy.candidate_limit, self._lexical_index.max_candidates)
        )
        passages = fuse_candidates(
            dense,
            lexical,
            corpus_version_id=self._candidate.corpus_version_id,
            candidate_limit=self._strategy.candidate_limit,
            rrf_k=self._strategy.rrf_k,
        )
        if self._strategy.reranker is not None:
            try:
                passages = _validate_reranked(
                    self._strategy.reranker.rerank(question, passages),
                    corpus_version_id=self._candidate.corpus_version_id,
                    candidate_limit=self._strategy.candidate_limit,
                )
            except Exception as exc:
                raise HybridRetrievalError("selected reranker failed") from exc
        passages = passages[: self._strategy.output_limit]
        return RetrievalResult(
            candidate=self._candidate,
            ordered_chunk_ids=[passage.chunk_id for passage in passages],
            scores=[passage.relevance_score for passage in passages if passage.relevance_score is not None],
            score_semantics="calibrated",
            strategy="reranked" if self._strategy.reranker is not None else "hybrid",
            stage_status="succeeded",
            search_ready=True,
        )


def fuse_candidates(
    dense: Iterable[RetrievedPassage],
    lexical: Iterable[LexicalCandidate],
    *,
    corpus_version_id: str,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
) -> tuple[RetrievedPassage, ...]:
    """Fuse two ranked lists by reciprocal rank, preserving safe provenance."""

    if not corpus_version_id.strip():
        raise HybridRetrievalError("corpus_version_id must be non-empty")
    if candidate_limit < 1 or rrf_k < 1:
        raise HybridRetrievalError("candidate_limit and rrf_k must be positive")

    merged: dict[str, tuple[float, RetrievedPassage]] = {}
    dense_ids: set[str] = set()
    for rank, passage in enumerate(dense, start=1):
        _require_corpus(passage.chunk_id, corpus_version_id)
        if passage.chunk_id in dense_ids:
            continue
        dense_ids.add(passage.chunk_id)
        if passage.relevance_score is None or not isfinite(passage.relevance_score):
            raise HybridRetrievalError("dense candidate score must be finite")
        score = 1 / (rrf_k + rank)
        merged[passage.chunk_id] = (
            score,
            passage.model_copy(update={"relevance_score": score}),
        )
    lexical_ids: set[str] = set()
    for rank, candidate in enumerate(lexical, start=1):
        _require_corpus(candidate.chunk_id, corpus_version_id)
        if candidate.chunk_id in lexical_ids:
            continue
        lexical_ids.add(candidate.chunk_id)
        score = 1 / (rrf_k + rank)
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = (
                score,
                RetrievedPassage(
                    chunk_id=candidate.chunk_id,
                    page=candidate.page,
                    section=candidate.section,
                    excerpt=candidate.excerpt,
                    relevance_score=score,
                ),
            )
        else:
            fused_score = existing[0] + score
            merged[candidate.chunk_id] = (
                fused_score,
                existing[1].model_copy(update={"relevance_score": fused_score}),
            )
    ranked = sorted(merged.values(), key=lambda item: (-item[0], item[1].chunk_id))
    return tuple(item[1] for item in ranked[:candidate_limit])


def _validate_reranked(
    candidates: Iterable[RetrievedPassage],
    *,
    corpus_version_id: str,
    candidate_limit: int,
) -> tuple[RetrievedPassage, ...]:
    validated = tuple(candidates)
    if len(validated) > candidate_limit:
        raise HybridRetrievalError("reranker exceeded the candidate limit")
    if len({candidate.chunk_id for candidate in validated}) != len(validated):
        raise HybridRetrievalError("reranker returned duplicate candidates")
    for candidate in validated:
        _require_corpus(candidate.chunk_id, corpus_version_id)
        if candidate.relevance_score is None or not isfinite(candidate.relevance_score):
            raise HybridRetrievalError("reranker returned an invalid score")
    return validated


def _require_corpus(chunk_id: str, corpus_version_id: str) -> None:
    if chunk_id.split(":", 1)[0] != corpus_version_id:
        raise HybridRetrievalError("candidate is outside the selected corpus")
