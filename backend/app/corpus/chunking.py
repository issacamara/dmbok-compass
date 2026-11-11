"""Deterministic, page-bounded text chunking for corpus ingestion."""

from __future__ import annotations

import csv
import hashlib
import io
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
PARENT_SEGMENTATION_VERSION = "semantic-parent-v2"
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
        ordinal = 0
        table_header: tuple[str, str] | None = None
        table_rows_seen = 0
        pending_prefix = parent.table_header or ""
        if parent.table_header:
            table_header = tuple((parent.table_header.splitlines() + [""])[:2])
            table_rows_seen = 2
        breadcrumb = _section_breadcrumb(parent.section_path)
        for span in _child_units(parent.source_spans, max_tokens):
            typed_table = span.block_kind == "table" and span.table_row_index is not None
            is_table_row = typed_table or span.text.lstrip().startswith("|")
            is_table_data = typed_table and span.table_row_index >= (span.table_header_row_count or 1)
            is_table_separator = is_table_row and (
                (typed_table and span.table_row_index < (span.table_header_row_count or 1))
                or bool(re.fullmatch(r"\s*\|?\s*[:|\- ]+\s*\|?\s*", span.text))
            )
            if not is_table_row:
                table_header, table_rows_seen = None, 0
            elif table_rows_seen == 0:
                if span.block_kind == "table":
                    table_header = tuple((span.table_header_text or span.text).splitlines()[:2])
                    table_rows_seen = span.table_header_row_count or 1
                else:
                    table_header, table_rows_seen = (span.text, ""), 1
            elif table_rows_seen == 1 and is_table_separator and table_header is not None:
                if span.block_kind != "table":
                    table_header = (table_header[0], span.text)
                table_rows_seen = 2
            else:
                table_rows_seen += 1
                if not typed_table:
                    is_table_data = table_rows_seen > 2
            candidate_text = "\n".join([breadcrumb] + ([pending_prefix] if pending_prefix else []) + [item.text for item in pending] + [span.text])
            if pending and _token_count(candidate_text) > target_tokens:
                children.append(_child(parent, ordinal, pending, synthetic_prefix=pending_prefix))
                ordinal += 1
                pending = []
                pending_prefix = (
                    span.table_header_text or "\n".join(table_header)
                    if is_table_row and is_table_data and table_header
                    else ""
                )
            elif not pending and is_table_row and is_table_data and table_header:
                pending_prefix = span.table_header_text or "\n".join(table_header)
            child_text = "\n".join([breadcrumb] + ([pending_prefix] if pending_prefix else []) + [item.text for item in pending] + [span.text])
            if _token_count(child_text) > max_tokens:
                raise ValueError("repeated table context and indivisible row exceed child maximum")
            pending.append(span)
        if pending:
            children.append(_child(parent, ordinal, pending, synthetic_prefix=pending_prefix))
    return tuple(children)


def _child(
    parent: ParentChunk,
    ordinal: int,
    spans: list[SourceSpan],
    *,
    synthetic_prefix: str = "",
) -> ChildDocumentChunk:
    source_text = "\n".join(span.text for span in spans)
    text = "\n".join(value for value in (_section_breadcrumb(parent.section_path), synthetic_prefix, source_text) if value)
    identity = "\x1f".join((parent.corpus_version_id, *parent.section_path, str(spans[0].page), str(spans[-1].page), str(ordinal), text))
    return ChildDocumentChunk(
        chunk_id=f"{parent.corpus_version_id}:child:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}",
        corpus_version_id=parent.corpus_version_id,
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=spans[0].page,
        page_end=spans[-1].page,
        source_spans=spans,
        source_text=source_text,
        child_ordinal=ordinal,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )


def _child_units(spans: Iterable[SourceSpan], max_tokens: int) -> tuple[SourceSpan, ...]:
    units: list[SourceSpan] = []
    for span in spans:
        if (span.block_kind == "table" and span.table_row_index is not None) or (
            span.block_kind == "list" and span.list_item_index is not None
        ):
            units.append(span)
            continue
        cursor = 0
        for line in span.text.splitlines(keepends=True):
            text = line.rstrip("\r\n")
            if text.strip():
                units.extend(_split_child_unit(span, span.start_offset + cursor, text, max_tokens))
            cursor += len(line)
    return tuple(units)


def _split_child_unit(source: SourceSpan, offset: int, text: str, max_tokens: int) -> tuple[SourceSpan, ...]:
    """Only prose may be split; Markdown rows/items remain intact."""
    if _token_count(text) <= max_tokens or text.lstrip().startswith(("|", "- ", "* ", "+ ")):
        return (_source_fragment(source, offset, text),)
    matches = tuple(re.finditer(r"\S+\s*", text))
    spans: list[SourceSpan] = []
    start = 0
    for index, match in enumerate(matches):
        end = match.end()
        if index > 0 and _token_count(text[start:end].rstrip()) > max_tokens:
            boundary = matches[index - 1].end()
            value = text[start:boundary].rstrip()
            spans.append(_source_fragment(source, offset + start, value))
            start = matches[index].start()
    value = text[start:].rstrip()
    if value:
        spans.append(_source_fragment(source, offset + start, value))
    return tuple(spans)


def _source_fragment(
    source: SourceSpan,
    start_offset: int,
    text: str,
    *,
    table_row_index: int | None = None,
    table_header_text: str | None = None,
    table_header_row_count: int | None = None,
    list_item_index: int | None = None,
) -> SourceSpan:
    end_offset = start_offset + len(text)
    span_id = source.span_id if start_offset == source.start_offset and end_offset == source.end_offset else None
    if source.block_id is not None and span_id is None:
        span_id = hashlib.sha256(
            f"{source.block_id}:{start_offset}:{end_offset}".encode("utf-8")
        ).hexdigest()[:24]
    return SourceSpan(
        page=source.page,
        start_offset=start_offset,
        end_offset=end_offset,
        text=text,
        span_id=span_id,
        block_id=source.block_id,
        block_order=source.block_order,
        printed_page_label=source.printed_page_label,
        block_kind=source.block_kind,
        table_row_index=table_row_index,
        table_header_text=table_header_text,
        table_header_row_count=table_header_row_count,
        list_item_index=list_item_index,
    )


def _section_breadcrumb(section_path: tuple[str, ...]) -> str:
    return "Section: " + " > ".join(section_path)


def build_llamaindex_nodes(
    parents: Iterable[ParentChunk], children: Iterable[ChildDocumentChunk]
) -> tuple[object, ...]:
    """Materialize the validated hierarchy as linked LlamaIndex nodes."""

    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    parent_records = tuple(parents)
    child_records = tuple(children)
    parent_nodes = {
        parent.parent_id: TextNode(
            id_=parent.parent_id,
            text=parent.text,
            metadata={
                "section_path": list(parent.section_path),
                "page_start": parent.page_start,
                "page_end": parent.page_end,
                "source_text": parent.source_text,
                "source_spans": [span.model_dump(mode="json", exclude_none=True) for span in parent.source_spans],
            },
        )
        for parent in parent_records
    }
    child_nodes = []
    children_by_parent: dict[str, list[RelatedNodeInfo]] = {parent_id: [] for parent_id in parent_nodes}
    for child in child_records:
        if child.parent_id not in parent_nodes:
            raise ValueError("LlamaIndex child node references a missing parent")
        node = TextNode(
            id_=child.chunk_id,
            text=child.text,
            metadata={
                "parent_id": child.parent_id,
                "section_path": list(child.section_path),
                "page_start": child.page_start,
                "page_end": child.page_end,
                "source_text": child.source_text,
                "source_spans": [span.model_dump(mode="json", exclude_none=True) for span in child.source_spans],
            },
            relationships={
                NodeRelationship.PARENT: RelatedNodeInfo(node_id=child.parent_id),
            },
        )
        child_nodes.append(node)
        children_by_parent[child.parent_id].append(RelatedNodeInfo(node_id=child.chunk_id))
    for parent_id, node in parent_nodes.items():
        node.relationships[NodeRelationship.CHILD] = children_by_parent[parent_id]
    return (*parent_nodes.values(), *child_nodes)


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
        breadcrumb = _section_breadcrumb(section.section_path)
        units = _section_units(
            section,
            max_tokens=max(1, max_tokens - _token_count(breadcrumb)),
        )
        pending: list[SourceSpan] = []
        parent_ordinal = 0
        table_header: list[str] = []
        table_rows_seen = 0
        pending_table_header: str | None = None
        for unit in units:
            unit_tokens = _token_count(unit.text)
            typed_table = unit.block_kind == "table" and unit.table_row_index is not None
            is_table_unit = typed_table or unit.text.lstrip().startswith("|")
            is_separator = is_table_unit and (
                (typed_table and unit.table_row_index < (unit.table_header_row_count or 1))
                or bool(re.fullmatch(r"\s*\|?\s*[:|\- ]+\s*\|?\s*", unit.text))
            )
            table_header_text = unit.table_header_text or "\n".join(table_header)
            is_table_data = typed_table and unit.table_row_index >= (unit.table_header_row_count or 1)
            if is_table_unit:
                if typed_table:
                    table_header = (unit.table_header_text or unit.text).splitlines()
                    table_rows_seen = unit.table_row_index + 1
                else:
                    table_header_text = "\n".join(table_header)
                    if table_rows_seen == 0:
                        table_header, table_rows_seen = [unit.text], 1
                    elif table_rows_seen == 1 and is_separator:
                        table_header.append(unit.text)
                        table_rows_seen = 2
                    else:
                        table_rows_seen += 1
                        is_table_data = table_rows_seen > 2
                table_header_text = unit.table_header_text or "\n".join(table_header)
                if is_table_data:
                    combined_table_tokens = _token_count(f"{table_header_text}\n{unit.text}")
                    if combined_table_tokens > max_tokens:
                        exceptions.append(
                            ParentReviewException(
                                section.section_path, unit.page, unit.page,
                                combined_table_tokens,
                                "repeated table header and row exceed parent maximum",
                            )
                        )
                        continue
            else:
                table_header, table_rows_seen = [], 0
            if unit_tokens > max_tokens:
                if pending:
                    parents.append(_parent(
                        corpus_version_id, section, section_ordinal, parent_ordinal, pending,
                        repeated_table_header=pending_table_header,
                    ))
                    parent_ordinal += 1
                    pending = []
                    pending_table_header = None
                exceptions.append(
                    ParentReviewException(
                        section.section_path, unit.page, unit.page, unit_tokens,
                        "indivisible list item or table row exceeds parent maximum",
                    )
                )
                continue
            candidate_text = "\n".join([breadcrumb] + ([pending_table_header] if pending_table_header else []) + [item.text for item in pending] + [unit.text])
            if pending and _token_count(candidate_text) > target_tokens:
                parents.append(_parent(
                    corpus_version_id, section, section_ordinal, parent_ordinal, pending,
                    repeated_table_header=pending_table_header,
                ))
                parent_ordinal += 1
                pending = []
                pending_table_header = None
                if is_table_data:
                    pending_table_header = table_header_text
            single_text = "\n".join(value for value in (breadcrumb, pending_table_header, unit.text) if value)
            if _token_count(single_text) > max_tokens:
                exceptions.append(ParentReviewException(
                    section.section_path, unit.page, unit.page, _token_count(single_text),
                    "structural context and indivisible source unit exceed parent maximum",
                ))
                continue
            pending.append(unit)
        if pending:
            parents.append(_parent(
                corpus_version_id, section, section_ordinal, parent_ordinal, pending,
                repeated_table_header=pending_table_header,
            ))
    return ParentSegmentation(tuple(parents), tuple(exceptions))


def _parent(
    corpus_version_id: str,
    section: LogicalSection,
    section_ordinal: int,
    parent_ordinal: int,
    spans: list[SourceSpan],
    *,
    repeated_table_header: str | None = None,
) -> ParentChunk:
    source_text = "\n".join(span.text for span in spans)
    text = "\n".join(value for value in (_section_breadcrumb(section.section_path), repeated_table_header, source_text) if value)
    return ParentChunk(
        parent_id=f"{corpus_version_id}:parent:{section_ordinal}:{parent_ordinal}",
        corpus_version_id=corpus_version_id,
        section_path=section.section_path,
        page_start=spans[0].page,
        page_end=spans[-1].page,
        source_spans=spans,
        source_text=source_text,
        table_header=repeated_table_header,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )


def _section_units(section: LogicalSection, *, max_tokens: int) -> tuple[SourceSpan, ...]:
    """Use complete lines and list/table items, splitting only oversized prose."""

    units: list[SourceSpan] = []
    for span in section.source_spans:
        if span.block_kind == "table" and span.table_rows:
            units.extend(_table_units(span))
            continue
        if span.block_kind == "list" and span.list_items:
            units.extend(_list_units(span))
            continue
        cursor = 0
        for line in span.text.splitlines(keepends=True):
            text = line.rstrip("\r\n")
            if text.strip():
                units.extend(_split_child_unit(span, span.start_offset + cursor, text, max_tokens))
            cursor += len(line)
    return tuple(units)


def _table_units(source: SourceSpan) -> tuple[SourceSpan, ...]:
    """Turn a parsed table into atomic raw row spans, including quoted CSV records."""

    if source.text.lstrip().startswith("|"):
        rows: list[tuple[int, int, str]] = []
        cursor = 0
        for line in source.text.splitlines(keepends=True):
            text = line.rstrip("\r\n")
            if text.strip():
                rows.append((cursor, cursor + len(text), text))
            cursor += len(line)
        if not rows:
            raise ValueError("parsed Markdown table has no source rows")
        header_count = 2 if len(rows) > 1 and re.fullmatch(r"\s*\|?\s*[:|\- ]+\s*\|?\s*", rows[1][2]) else 1
        markdown_rows = [
            _markdown_table_cells(row[2])
            for index, row in enumerate(rows)
            if not (index == 1 and header_count == 2)
        ]
        if markdown_rows != list(source.table_rows):
            raise ValueError("parsed Markdown table rows do not match their raw source rows")
        header_start = rows[0][0]
        header_end = rows[header_count - 1][1]
        header_text = source.text[header_start:header_end]
        units = [
            _source_fragment(
                source,
                source.start_offset + header_start,
                header_text,
                table_row_index=0,
                table_header_text=header_text,
                table_header_row_count=header_count,
            )
        ]
        units.extend(
            _source_fragment(
                source,
                source.start_offset + start,
                text,
                table_row_index=index,
                table_header_text=header_text,
                table_header_row_count=header_count,
            )
            for index, (start, _, text) in enumerate(rows[header_count:], start=header_count)
        )
        return tuple(units)

    line_starts = [0]
    for line in source.text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    reader = csv.reader(io.StringIO(source.text, newline=""))
    records: list[tuple[int, int]] = []
    parsed_rows: list[tuple[str, ...]] = []
    previous_line = 0
    for row in reader:
        end_line = reader.line_num
        records.append((line_starts[previous_line], line_starts[end_line]))
        parsed_rows.append(tuple(row))
        previous_line = end_line
    if parsed_rows != list(source.table_rows):
        raise ValueError("parsed CSV table rows do not match their raw source records")
    raw_records = [source.text[start:end].rstrip("\r\n") for start, end in records]
    if not raw_records or any(not record for record in raw_records):
        raise ValueError("parsed CSV table has an empty source record")
    header_text = raw_records[0]
    return tuple(
        _source_fragment(
            source,
            source.start_offset + start,
            record,
            table_row_index=index,
            table_header_text=header_text,
            table_header_row_count=1,
        )
        for index, ((start, _), record) in enumerate(zip(records, raw_records, strict=True))
    )


def _markdown_table_cells(row: str) -> tuple[str, ...]:
    content = row.strip()
    if content.startswith("|"):
        content = content[1:]
    if content.endswith("|") and not content.endswith("\\|"):
        content = content[:-1]
    return tuple(cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", content))


def _list_units(source: SourceSpan) -> tuple[SourceSpan, ...]:
    """Preserve each parser-reviewed list item as one indivisible source unit."""

    units: list[SourceSpan] = []
    cursor = 0
    for index, item in enumerate(source.list_items):
        start = source.text.find(item, cursor)
        if start < 0:
            raise ValueError("parsed list item cannot be located in its source block")
        line_start = source.text.rfind("\n", 0, start) + 1
        end = start + len(item)
        units.append(
            _source_fragment(
                source,
                source.start_offset + line_start,
                source.text[line_start:end],
                list_item_index=index,
            )
        )
        cursor = end
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
