from __future__ import annotations

from hashlib import sha256

import pytest

from app.answering.citations import CitationValidationError, validated_citations
from app.contracts import (
    ChildDocumentChunk, Citation, GenerationContext, GenerationContextParent,
    ParentChunk, RawSourceSpan, RetrievedPassage, SourceSpan,
)
from model.protocol import GenerationCandidate


def source_parent(
    source_spans: list[SourceSpan], *, parent_id: str = "v1:parent:0", corpus: str = "v1"
) -> ParentChunk:
    return ParentChunk(
        parent_id=parent_id,
        corpus_version_id=corpus,
        section_path=("Data Management", "Governance"),
        page_start=source_spans[0].page,
        page_end=source_spans[-1].page,
        source_spans=source_spans,
        content_hash=sha256("\n".join(span.text for span in source_spans).encode()).hexdigest(),
        token_count=10,
        text="\n".join(span.text for span in source_spans),
    )


def source_child(
    spans: list[SourceSpan], *, chunk_id: str = "v1:child:0", parent_id: str = "v1:parent:0", corpus: str = "v1"
) -> ChildDocumentChunk:
    return ChildDocumentChunk(
        chunk_id=chunk_id,
        corpus_version_id=corpus,
        parent_id=parent_id,
        section_path=("Data Management", "Governance"),
        page_start=spans[0].page,
        page_end=spans[-1].page,
        source_spans=spans,
        child_ordinal=0,
        content_hash=sha256("\n".join(span.text for span in spans).encode()).hexdigest(),
        token_count=10,
        text="\n".join(span.text for span in spans),
    )


def candidate(citation_id: str, excerpt: str, *, page: int = 1, section: str = "wrong") -> GenerationCandidate:
    return GenerationCandidate(
        outcome="answer",
        answer_text="A grounded claim.",
        citations=[{
            "citation_id": citation_id,
            "page": page,
            "section": section,
            "excerpt": excerpt,
        }],
    )


def retrieved(child: ChildDocumentChunk) -> tuple[RetrievedPassage, ...]:
    return (RetrievedPassage(
        chunk_id=child.chunk_id,
        page=child.page_start,
        section=" / ".join(child.section_path),
        excerpt=child.text,
        relevance_score=.9,
    ),)


def test_citations_reconstruct_page_section_and_excerpt_from_child_source_spans() -> None:
    first = SourceSpan(
        page=1, start_offset=10, end_offset=21, text="First page.",
        span_id="span-first", block_id="block-first", block_order=4,
    )
    second = SourceSpan(
        page=2, start_offset=30, end_offset=46, text="Second page text",
        span_id="span-second", block_id="block-second", block_order=5,
        printed_page_label="xiv",
    )
    parent = source_parent([first, second])
    child = source_child([first, second])

    citations = validated_citations(
        candidate(child.chunk_id, "Second page", page=1, section="Untrusted section"),
        retrieved(child),
        child_sources=(child,),
        source_parents=(parent,),
        corpus_version_id="v1",
    )

    assert len(citations) == 1
    assert citations[0].page == 2
    assert citations[0].section == "Data Management / Governance"
    assert citations[0].excerpt == "Second page"


def test_cross_child_citations_keep_each_child_page_and_source_excerpt() -> None:
    first = SourceSpan(
        page=1, start_offset=0, end_offset=16, text="First child span",
        span_id="span-first", block_id="block-first",
    )
    second = SourceSpan(
        page=2, start_offset=0, end_offset=17, text="Second child span",
        span_id="span-second", block_id="block-second",
    )
    parent = source_parent([first, second])
    first_child = source_child([first], chunk_id="v1:child:first")
    second_child = source_child([second], chunk_id="v1:child:second")
    context = GenerationContext(
        corpus_version_id="v1",
        parents=(GenerationContextParent(
            parent_id=parent.parent_id,
            spans=(
                RawSourceSpan(
                    span_id="span-first", block_id="block-first", physical_page=1,
                    start_offset=0, end_offset=16, raw_text=first.text,
                ),
                RawSourceSpan(
                    span_id="span-second", block_id="block-second", physical_page=2,
                    start_offset=0, end_offset=17, raw_text=second.text,
                ),
            ),
            backing_child_ids=(first_child.chunk_id, second_child.chunk_id),
            embedding_text=parent.text,
        ),),
    )
    answer = GenerationCandidate(
        outcome="answer",
        answer_text="A claim supported across children.",
        citations=[
            {"citation_id": first_child.chunk_id, "page": 9, "section": "wrong", "excerpt": first.text},
            {"citation_id": second_child.chunk_id, "page": 9, "section": "wrong", "excerpt": second.text},
        ],
    )

    citations = validated_citations(
        answer,
        retrieved(first_child) + retrieved(second_child),
        child_sources=(first_child, second_child),
        source_parents=(parent,),
        generation_context=context,
        corpus_version_id="v1",
    )

    assert [(item.page, item.section, item.excerpt) for item in citations] == [
        (1, "Data Management / Governance", first.text),
        (2, "Data Management / Governance", second.text),
    ]
    assert all(item.excerpt in {first.text, second.text} and len(item.excerpt) <= 500 for item in citations)


def test_context_allowlist_preserves_physical_page_and_table_header_source_text() -> None:
    text = "Column | Value\nMetric | 42"
    span = SourceSpan(
        page=25, start_offset=100, end_offset=100 + len(text), text=text,
        span_id="table-span", block_id="table-block", block_order=3,
        block_kind="table", table_header_text="Column | Value", table_header_row_count=1,
        table_row_index=1,
    )
    parent = source_parent([span], parent_id="v1:parent:table")
    child = source_child([span], chunk_id="v1:child:table", parent_id=parent.parent_id)
    context = GenerationContext(
        corpus_version_id="v1",
        parents=(GenerationContextParent(
            parent_id=parent.parent_id,
            spans=(RawSourceSpan(
                span_id=span.span_id, block_id=span.block_id,
                physical_page=span.page, printed_page_label=span.printed_page_label,
                start_offset=span.start_offset, end_offset=span.end_offset,
                raw_text=span.text,
            ),),
            backing_child_ids=(child.chunk_id,),
            embedding_text=parent.text,
        ),),
    )

    citation = validated_citations(
        candidate(child.chunk_id, "Metric | 42", page=1),
        retrieved(child), child_sources=(child,), source_parents=(parent,),
        generation_context=context, corpus_version_id="v1",
    )[0]

    assert citation.page == 25
    assert citation.excerpt == "Metric | 42"
    assert len(citation.excerpt) <= 500


def test_citation_cannot_use_a_ranked_child_outside_supplied_context_allowlist() -> None:
    first_span = SourceSpan(
        page=1, start_offset=0, end_offset=10, text="First text",
        span_id="span-first", block_id="block-first",
    )
    second_span = SourceSpan(
        page=2, start_offset=0, end_offset=11, text="Second text",
        span_id="span-second", block_id="block-second",
    )
    first_parent = source_parent([first_span], parent_id="v1:parent:first")
    second_parent = source_parent([second_span], parent_id="v1:parent:second")
    first_child = source_child([first_span], chunk_id="v1:child:first", parent_id=first_parent.parent_id)
    second_child = source_child([second_span], chunk_id="v1:child:second", parent_id=second_parent.parent_id)
    context = GenerationContext(
        corpus_version_id="v1",
        parents=(GenerationContextParent(
            parent_id=first_parent.parent_id,
            spans=(RawSourceSpan(
                span_id="span-first", block_id="block-first", physical_page=1,
                start_offset=0, end_offset=10, raw_text="First text",
            ),),
            backing_child_ids=(first_child.chunk_id,),
            embedding_text=first_parent.text,
        ),),
    )

    with pytest.raises(CitationValidationError, match="outside supplied generation context"):
        validated_citations(
            candidate(second_child.chunk_id, second_span.text),
            retrieved(first_child) + retrieved(second_child),
            child_sources=(first_child, second_child),
            source_parents=(first_parent, second_parent),
            generation_context=context,
            corpus_version_id="v1",
        )


def test_citations_reject_parent_only_unknown_and_missing_parent_evidence() -> None:
    span = SourceSpan(page=1, start_offset=0, end_offset=11, text="Source text")
    parent = source_parent([span])
    child = source_child([span])

    with pytest.raises(CitationValidationError, match="retrieved child set"):
        validated_citations(
            candidate(parent.parent_id, "Source"), retrieved(child),
            child_sources=(child,), source_parents=(parent,), corpus_version_id="v1",
        )
    with pytest.raises(CitationValidationError, match="no request-local parent"):
        validated_citations(
            candidate(child.chunk_id, "Source"), retrieved(child),
            child_sources=(child,), source_parents=(), corpus_version_id="v1",
        )


def test_citations_reject_cross_corpus_and_altered_source_spans() -> None:
    span = SourceSpan(page=1, start_offset=0, end_offset=11, text="Source text")
    parent = source_parent([span])
    child = source_child([span])

    with pytest.raises(CitationValidationError, match="crosses corpus"):
        validated_citations(
            candidate(child.chunk_id, "Source"), retrieved(child),
            child_sources=(child,), source_parents=(parent,), corpus_version_id="v2",
        )
    with pytest.raises(CitationValidationError, match="exact source span"):
        validated_citations(
            candidate(child.chunk_id, "Altered source"), retrieved(child),
            child_sources=(child,), source_parents=(parent,), corpus_version_id="v1",
        )


def test_citations_reject_a_child_span_not_contained_by_its_parent() -> None:
    parent = source_parent([SourceSpan(page=1, start_offset=0, end_offset=11, text="Other text")])
    child = source_child([SourceSpan(page=1, start_offset=0, end_offset=11, text="Source text")])

    with pytest.raises(CitationValidationError, match="source text or offsets changed"):
        validated_citations(
            candidate(child.chunk_id, "Source"), retrieved(child),
            child_sources=(child,), source_parents=(parent,), corpus_version_id="v1",
        )


@pytest.mark.parametrize("stale_record", ["parent", "child"])
def test_citations_reject_coherently_changed_text_with_stale_hash(
    stale_record: str,
) -> None:
    original = SourceSpan(page=1, start_offset=0, end_offset=13, text="Original data")
    tampered = SourceSpan(page=1, start_offset=0, end_offset=13, text="Tampered data")
    original_parent = source_parent([original])
    original_child = source_child([original])
    parent = source_parent([tampered])
    child = source_child([tampered])
    if stale_record == "parent":
        parent = parent.model_copy(update={"content_hash": original_parent.content_hash})
    else:
        child = child.model_copy(update={"content_hash": original_child.content_hash})

    with pytest.raises(CitationValidationError, match="content hash changed"):
        validated_citations(
            candidate(child.chunk_id, tampered.text),
            retrieved(child),
            child_sources=(child,),
            source_parents=(parent,),
            corpus_version_id="v1",
        )


def test_public_citation_excerpt_is_bounded_to_500_characters() -> None:
    text = "x" * 600
    span = SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)
    parent = source_parent([span])
    child = source_child([span])

    citation = validated_citations(
        candidate(child.chunk_id, text), retrieved(child),
        child_sources=(child,), source_parents=(parent,), corpus_version_id="v1",
    )[0]

    assert citation.excerpt == text[:500]


def test_public_citation_contract_bounds_optional_excerpt() -> None:
    assert Citation(citation_id="c1", page=1, section="S", excerpt="x" * 500).excerpt == "x" * 500
    with pytest.raises(ValueError, match="at most 500 characters"):
        Citation(citation_id="c1", page=1, section="S", excerpt="x" * 501)
