"""Bind compatible legacy evaluation passages to immutable parsed evidence."""

from __future__ import annotations

from dataclasses import dataclass

from app.contracts import ChildDocumentChunk, EvaluationRowV2, ParentChunk, RawSourceSpan, SourceSpan


@dataclass(frozen=True)
class ImportBindingIssue:
    index: int
    code: str
    message: str


@dataclass(frozen=True)
class BoundLegacyImport:
    index: int
    row: EvaluationRowV2


def bind_legacy_imports(
    records: tuple[tuple[int, str, str, str, str, str], ...],
    *,
    parents: tuple[ParentChunk, ...],
    children: tuple[ChildDocumentChunk, ...],
) -> tuple[tuple[BoundLegacyImport, ...], tuple[ImportBindingIssue, ...]]:
    """Bind a passage only when it occurs once within one parse-derived source span.

    Each record tuple is (index, item_id, question, category, rubric, passage).
    A partial exact passage is allowed, but normalization, synthetic identities,
    cross-span stitching, and ambiguous occurrences are deliberately rejected.
    """
    parent_by_id = {parent.parent_id: parent for parent in parents}
    corpus_id = parents[0].corpus_version_id if parents else None
    bound: list[BoundLegacyImport] = []
    issues: list[ImportBindingIssue] = []

    for index, item_id, question, category, rubric, passage in records:
        matches: list[tuple[ChildDocumentChunk, SourceSpan, int, int]] = []
        if passage:
            for child in children:
                if child.corpus_version_id != corpus_id:
                    continue
                parent = parent_by_id.get(child.parent_id)
                if parent is None:
                    continue
                for span in child.source_spans:
                    if span.span_id is None or span.block_id is None:
                        continue
                    start = span.text.find(passage)
                    while start >= 0:
                        matches.append((child, span, start, start + len(passage)))
                        start = span.text.find(passage, start + 1)

        if len(matches) != 1:
            code = "ambiguous_supporting_passage" if matches else "unmatched_supporting_passage"
            message = (
                "Passage matches multiple parse-derived source locations; revise it to identify one exact span."
                if matches
                else "Passage did not exactly match a parse-derived source span in the selected corpus."
            )
            issues.append(ImportBindingIssue(index=index, code=code, message=message))
            continue

        child, span, start, end = matches[0]
        parent = parent_by_id[child.parent_id]
        raw_span = RawSourceSpan(
            span_id=span.span_id or "",
            block_id=span.block_id or "",
            physical_page=span.page,
            printed_page_label=span.printed_page_label,
            start_offset=span.start_offset + start,
            end_offset=span.start_offset + end,
            raw_text=span.text[start:end],
        )
        try:
            row = EvaluationRowV2.model_validate({
                "item_id": item_id,
                "origin": "imported",
                "question": question,
                "category": category,
                "expected_outcome": "answer",
                "expected_answer_or_rubric": rubric,
                "expected_refusal": False,
                "source_spans": (raw_span,),
                "relevant_child_ids": (child.chunk_id,),
                "parent_ids": (parent.parent_id,),
            })
        except (TypeError, ValueError):
            issues.append(ImportBindingIssue(
                index=index,
                code="invalid_row_v2",
                message="Passage matched, but the imported row did not satisfy the row-v2 contract.",
            ))
            continue
        bound.append(BoundLegacyImport(index=index, row=row))
    return tuple(bound), tuple(issues)
