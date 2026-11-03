from __future__ import annotations

import pytest

from app.answering.citations import CitationValidationError, validated_citations
from app.contracts import ChildDocumentChunk, Citation, ParentChunk, RetrievedPassage, SourceSpan
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
        content_hash="a" * 64,
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
        content_hash="b" * 64,
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
    first = SourceSpan(page=1, start_offset=10, end_offset=21, text="First page.")
    second = SourceSpan(page=2, start_offset=30, end_offset=46, text="Second page text")
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
    first = SourceSpan(page=1, start_offset=0, end_offset=16, text="First child span")
    second = SourceSpan(page=2, start_offset=0, end_offset=17, text="Second child span")
    parent = source_parent([first, second])
    first_child = source_child([first], chunk_id="v1:child:first")
    second_child = source_child([second], chunk_id="v1:child:second")
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
        corpus_version_id="v1",
    )

    assert [(item.page, item.section, item.excerpt) for item in citations] == [
        (1, "Data Management / Governance", first.text),
        (2, "Data Management / Governance", second.text),
    ]


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
