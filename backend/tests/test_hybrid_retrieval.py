from __future__ import annotations

import pytest

from app.contracts import CandidateIdentity, RetrievedPassage
from app.retrieval.hybrid import (
    HybridRetrievalError,
    HybridRetriever,
    HybridStrategy,
    fuse_candidates,
)
from app.retrieval.lexical import LexicalCandidate, LexicalDocument, LexicalIndex


def passage(chunk_id: str, score: float) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=1,
        section="Section",
        excerpt=f"Text for {chunk_id}",
        relevance_score=score,
    )


def candidate() -> CandidateIdentity:
    return CandidateIdentity(
        artifact_id="candidate-artifact",
        installed_lock_sha256="a" * 64,
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )


def test_fusion_deduplicates_ranks_deterministically_and_caps_candidates() -> None:
    dense = tuple(passage(f"corpus-v1:{index}:0", score) for index, score in enumerate((0.9, 0.8, 0.7)))
    lexical = tuple(
        LexicalCandidate(f"corpus-v1:{index}:0", 10 - index, 1, "Section", "lexical")
        for index in range(1, 8)
    )

    result = fuse_candidates(dense, lexical, corpus_version_id="corpus-v1", candidate_limit=5)

    assert len(result) == 5
    assert len({item.chunk_id for item in result}) == 5
    assert result[0].chunk_id == "corpus-v1:1:0"
    assert all(item.relevance_score is not None for item in result)


def test_fusion_rejects_candidates_from_another_corpus() -> None:
    with pytest.raises(HybridRetrievalError, match="outside the selected corpus"):
        fuse_candidates(
            (passage("corpus-v2:1:0", 0.9),),
            (),
            corpus_version_id="corpus-v1",
        )


def test_hybrid_result_uses_calibrated_strategy_scores() -> None:
    index = LexicalIndex(
        (LexicalDocument("corpus-v1:2:0", 1, "Section", "exact terminology", ("exact", "terminology")),),
    )

    def dense(question: str, *, active_corpus_version_id: str, limit: int):
        assert active_corpus_version_id == "corpus-v1"
        return (passage("corpus-v1:1:0", 0.9),)

    result = HybridRetriever(dense, index, candidate()).retrieve("exact terminology")

    assert result.strategy == "hybrid"
    assert result.score_semantics == "calibrated"
    assert result.ordered_chunk_ids == ["corpus-v1:1:0", "corpus-v1:2:0"]
    assert result.search_ready is True


def test_reranker_is_bounded_and_explicitly_selected() -> None:
    class Reranker:
        def rerank(self, question, candidates):
            return tuple(reversed(candidates))

    index = LexicalIndex(
        (LexicalDocument("corpus-v1:2:0", 1, "Section", "exact", ("exact",)),),
    )

    def dense(question: str, *, active_corpus_version_id: str, limit: int):
        return (passage("corpus-v1:1:0", 0.9),)

    result = HybridRetriever(
        dense, index, candidate(), strategy=HybridStrategy(reranker=Reranker())
    ).retrieve("exact")

    assert result.strategy == "reranked"
    assert result.ordered_chunk_ids == ["corpus-v1:2:0", "corpus-v1:1:0"]


def test_reranker_failure_fails_closed() -> None:
    class BrokenReranker:
        def rerank(self, question, candidates):
            raise RuntimeError("provider unavailable")

    index = LexicalIndex(
        (LexicalDocument("corpus-v1:1:0", 1, "Section", "exact", ("exact",)),),
    )

    with pytest.raises(HybridRetrievalError, match="selected reranker failed"):
        HybridRetriever(
            lambda *args, **kwargs: (passage("corpus-v1:1:0", 0.9),),
            index,
            candidate(),
            strategy=HybridStrategy(reranker=BrokenReranker()),
        ).retrieve("exact")
