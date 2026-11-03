"""Deterministic, page-bounded text chunking for corpus ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from app.contracts import ChildDocumentChunk, ParentChunk, SourceSpan
from app.corpus.structure_map import LogicalSection


_WHITESPACE = re.compile(r"\s+")
MAX_CHUNK_CHARACTERS = 5_000
MAX_SECTION_CHARACTERS = 500
SUBDIVISION_ALGORITHM_VERSION = "fixed-character-window-v1"
PARENT_SEGMENTATION_VERSION = "semantic-parent-v1"
PARENT_TARGET_TOKENS = 1_000
PARENT_MAX_TOKENS = 1_500
CHILD_TARGET_TOKENS = 300
CHILD_MAX_TOKENS = 1_500


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


@dataclass(frozen=True, slots=True)
class ParentReviewException:
    """An intact source item that is too large to safely segment automatically."""

    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    token_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class ParentSegmentation:
    parents: tuple[ParentChunk, ...]
    exceptions: tuple[ParentReviewException, ...]

    @property
    def ready_for_promotion(self) -> bool:
        return not self.exceptions


def segment_children(
    parents: Iterable[ParentChunk],
    *,
    target_tokens: int = CHILD_TARGET_TOKENS,
    max_tokens: int = CHILD_MAX_TOKENS,
) -> tuple[ChildDocumentChunk, ...]:
    """Split each parent into deterministic, non-overlapping embedding children."""

    if target_tokens < 1 or max_tokens < target_tokens:
        raise ValueError("child token limits must be positive and ordered")

    children: list[ChildDocumentChunk] = []
    for parent in parents:
        pending: list[SourceSpan] = []
        pending_tokens = 0
        ordinal = 0
        for span in _child_units(parent.source_spans, max_tokens):
            tokens = _token_count(span.text)
            if pending and pending_tokens + tokens > target_tokens:
                children.append(_child(parent, ordinal, pending))
                ordinal += 1
                pending, pending_tokens = [], 0
            pending.append(span)
            pending_tokens += tokens
        if pending:
            children.append(_child(parent, ordinal, pending))
    return tuple(children)


def _child(parent: ParentChunk, ordinal: int, spans: list[SourceSpan]) -> ChildDocumentChunk:
    text = "\n".join(span.text for span in spans)
    identity = "\x1f".join((parent.corpus_version_id, *parent.section_path, str(spans[0].page), str(spans[-1].page), str(ordinal), text))
    return ChildDocumentChunk(
        chunk_id=f"{parent.corpus_version_id}:child:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}",
        corpus_version_id=parent.corpus_version_id,
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=spans[0].page,
        page_end=spans[-1].page,
        source_spans=spans,
        child_ordinal=ordinal,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )


def _child_units(spans: Iterable[SourceSpan], max_tokens: int) -> tuple[SourceSpan, ...]:
    units: list[SourceSpan] = []
    for span in spans:
        cursor = 0
        for line in span.text.splitlines(keepends=True):
            text = line.rstrip("\r\n")
            if text.strip():
                units.extend(_split_child_unit(span.page, span.start_offset + cursor, text, max_tokens))
            cursor += len(line)
    return tuple(units)


def _split_child_unit(page: int, offset: int, text: str, max_tokens: int) -> tuple[SourceSpan, ...]:
    """Only prose may be split; Markdown rows/items remain intact."""
    if _token_count(text) <= max_tokens or text.lstrip().startswith(("|", "- ", "* ", "+ ")):
        return (SourceSpan(page=page, start_offset=offset, end_offset=offset + len(text), text=text),)
    matches = tuple(re.finditer(r"\S+\s*", text))
    return tuple(
        SourceSpan(page=page, start_offset=offset + group[0].start(), end_offset=offset + group[-1].end(), text=text[group[0].start() : group[-1].end()].rstrip())
        for group in (matches[index : index + max_tokens] for index in range(0, len(matches), max_tokens))
    )


def segment_parents(
    sections: Iterable[LogicalSection],
    *,
    corpus_version_id: str,
    target_tokens: int = PARENT_TARGET_TOKENS,
    max_tokens: int = PARENT_MAX_TOKENS,
) -> ParentSegmentation:
    """Create bounded parent records without splitting a source list item or table row.

    This is additive while child-chunk migration is in flight; ``chunk_pages``
    remains only for the legacy writer until issue #299 replaces that consumer.
    """

    if not corpus_version_id or corpus_version_id.strip() != corpus_version_id:
        raise ValueError("corpus_version_id must be a non-empty value without surrounding whitespace")
    if target_tokens < 1 or max_tokens < target_tokens:
        raise ValueError("parent token limits must be positive and ordered")

    parents: list[ParentChunk] = []
    exceptions: list[ParentReviewException] = []
    for section_ordinal, section in enumerate(sections):
        units = _section_units(section)
        pending: list[SourceSpan] = []
        pending_tokens = 0
        parent_ordinal = 0
        for unit in units:
            unit_tokens = _token_count(unit.text)
            if unit_tokens > max_tokens:
                if pending:
                    parents.append(_parent(corpus_version_id, section, section_ordinal, parent_ordinal, pending))
                    parent_ordinal += 1
                    pending, pending_tokens = [], 0
                exceptions.append(
                    ParentReviewException(
                        section.section_path, unit.page, unit.page, unit_tokens,
                        "indivisible list item or table row exceeds parent maximum",
                    )
                )
                continue
            if pending and pending_tokens + unit_tokens > target_tokens:
                parents.append(_parent(corpus_version_id, section, section_ordinal, parent_ordinal, pending))
                parent_ordinal += 1
                pending, pending_tokens = [], 0
            pending.append(unit)
            pending_tokens += unit_tokens
        if pending:
            parents.append(_parent(corpus_version_id, section, section_ordinal, parent_ordinal, pending))
    return ParentSegmentation(tuple(parents), tuple(exceptions))


def _parent(
    corpus_version_id: str,
    section: LogicalSection,
    section_ordinal: int,
    parent_ordinal: int,
    spans: list[SourceSpan],
) -> ParentChunk:
    text = "\n".join(span.text for span in spans)
    return ParentChunk(
        parent_id=f"{corpus_version_id}:parent:{section_ordinal}:{parent_ordinal}",
        corpus_version_id=corpus_version_id,
        section_path=section.section_path,
        page_start=spans[0].page,
        page_end=spans[-1].page,
        source_spans=spans,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )


def _section_units(section: LogicalSection) -> tuple[SourceSpan, ...]:
    """Use complete non-empty lines as the smallest safe structural boundary."""

    units: list[SourceSpan] = []
    for span in section.source_spans:
        cursor = 0
        for line in span.text.splitlines(keepends=True):
            text = line.rstrip("\r\n")
            if text.strip():
                units.append(SourceSpan(page=span.page, start_offset=span.start_offset + cursor, end_offset=span.start_offset + cursor + len(text), text=text))
            cursor += len(line)
    return tuple(units)


def _token_count(text: str) -> int:
    # Keep legacy extraction/proposal imports offline until parent segmentation is requested.
    from app.corpus.tokenization import count_tokens

    return count_tokens(text)


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
