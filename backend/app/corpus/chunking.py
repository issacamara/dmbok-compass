"""Deterministic, page-bounded text chunking for corpus ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SourcePage:
    """Text and provenance recovered from one source-PDF page."""

    page: int
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A deterministic chunk ready for embedding and persistence."""

    chunk_id: str
    corpus_version_id: str
    page: int
    section: str
    chunk_ordinal: int
    content_hash: str
    text: str


def chunk_pages(
    pages: Iterable[SourcePage],
    *,
    corpus_version_id: str,
    target_tokens: int = 700,
    overlap_tokens: int = 70,
) -> tuple[TextChunk, ...]:
    """Split pages into stable windows without crossing page boundaries.

    Tokens are whitespace-delimited words after line breaks and repeated
    whitespace have been normalized. Each page starts its ordinal at zero;
    therefore a chunk ID remains stable when unrelated pages are added.
    """

    if not corpus_version_id or corpus_version_id.strip() != corpus_version_id:
        raise ValueError("corpus_version_id must be a non-empty value without surrounding whitespace")
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than target_tokens")

    chunks: list[TextChunk] = []
    for source_page in pages:
        _validate_page(source_page)
        tokens = _normalized_tokens(source_page.text)
        if not tokens:
            continue

        step = target_tokens - overlap_tokens
        for ordinal, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + target_tokens]
            if not window:
                break
            text = " ".join(window)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=f"{corpus_version_id}:{source_page.page}:{ordinal}",
                    corpus_version_id=corpus_version_id,
                    page=source_page.page,
                    section=source_page.section.strip(),
                    chunk_ordinal=ordinal,
                    content_hash=content_hash,
                    text=text,
                )
            )
            if start + target_tokens >= len(tokens):
                break

    return tuple(chunks)


def _normalized_tokens(text: str) -> list[str]:
    return _WHITESPACE.sub(" ", text).strip().split(" ") if text.strip() else []


def _validate_page(source_page: SourcePage) -> None:
    if source_page.page < 1:
        raise ValueError("page must be positive")
    if not source_page.section or not source_page.section.strip():
        raise ValueError("section must be non-empty")
