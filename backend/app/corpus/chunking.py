"""Deterministic, page-bounded text chunking for corpus ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


_WHITESPACE = re.compile(r"\s+")
MAX_CHUNK_CHARACTERS = 5_000
MAX_SECTION_CHARACTERS = 500
SUBDIVISION_ALGORITHM_VERSION = "fixed-character-window-v1"


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

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=lambda text: len(text.split()),
    )
    chunks: list[TextChunk] = []
    for source_page in pages:
        _validate_page(source_page)
        normalized = " ".join(_normalized_tokens(source_page.text))
        if not normalized:
            continue
        documents = splitter.split_documents(
            [Document(page_content=normalized, metadata={"page": source_page.page})]
        )
        subdivisions = tuple(
            subdivision
            for document in documents
            for subdivision in _subdivide(document.page_content)
        )
        for ordinal, text in enumerate(subdivisions):
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
    return tuple(chunks)


def _normalized_tokens(text: str) -> list[str]:
    return _WHITESPACE.sub(" ", text).strip().split(" ") if text.strip() else []


def _validate_page(source_page: SourcePage) -> None:
    if source_page.page < 1:
        raise ValueError("page must be positive")
    if not source_page.section or not source_page.section.strip():
        raise ValueError("section must be non-empty")
    if len(source_page.section.strip()) > MAX_SECTION_CHARACTERS:
        raise ValueError(f"section must be at most {MAX_SECTION_CHARACTERS} characters")


def _subdivide(text: str) -> tuple[str, ...]:
    """Bound chunk text without dropping content or crossing source pages."""

    if not text:
        return ()
    return tuple(
        text[start : start + MAX_CHUNK_CHARACTERS]
        for start in range(0, len(text), MAX_CHUNK_CHARACTERS)
    )
