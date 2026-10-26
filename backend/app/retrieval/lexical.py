"""Deterministic, in-process lexical candidate generation.

This module is deliberately independent of Firestore and serving.  It reads a
validated artifact index and returns the same ranked candidates for the same
query and artifact.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


TOKENIZER_VERSION = "unicode-word-v1"
ALGORITHM_VERSION = "tf-idf-lite-v1"
_TOKEN = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    """A content-safe lexical retrieval result."""

    chunk_id: str
    score: float
    page: int
    section: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class LexicalDocument:
    chunk_id: str
    page: int
    section: str
    text: str
    terms: tuple[str, ...]


def tokenize(text: str, *, version: str = TOKENIZER_VERSION) -> tuple[str, ...]:
    if version != TOKENIZER_VERSION:
        raise ValueError(f"unsupported tokenizer version: {version}")
    return tuple(token.casefold() for token in _TOKEN.findall(text))


class LexicalIndex:
    """Bounded immutable index used by the optional lexical experiment."""

    def __init__(self, documents: Iterable[LexicalDocument], *, max_candidates: int = 5, max_cache_entries: int = 128) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if max_cache_entries < 1:
            raise ValueError("max_cache_entries must be positive")
        self._documents = tuple(documents)
        self.max_candidates = max_candidates
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[tuple[str, int], tuple[LexicalCandidate, ...]] = OrderedDict()
        self._document_frequency: dict[str, int] = {}
        for document in self._documents:
            for term in set(document.terms):
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

    def search(self, query: str, *, limit: int | None = None) -> tuple[LexicalCandidate, ...]:
        terms = tokenize(query)
        if not terms:
            return ()
        requested = self.max_candidates if limit is None else limit
        if requested < 1 or requested > self.max_candidates:
            raise ValueError(f"limit must be between 1 and {self.max_candidates}")
        cache_key = (query, requested)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        total = max(1, len(self._documents))
        query_terms = set(terms)
        candidates: list[LexicalCandidate] = []
        for document in self._documents:
            counts = {term: document.terms.count(term) for term in query_terms}
            score = sum(
                (1.0 + (count > 0))
                * (1.0 + __import__("math").log(total / self._document_frequency[term]))
                for term, count in counts.items()
                if count
            )
            if score:
                candidates.append(
                    LexicalCandidate(
                        chunk_id=document.chunk_id,
                        score=round(score, 12),
                        page=document.page,
                        section=document.section,
                        excerpt=document.text,
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        result = tuple(candidates[:requested])
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return result

    @property
    def cache_size(self) -> int:
        """Number of cached query results, bounded by ``max_cache_entries``."""

        return len(self._cache)
