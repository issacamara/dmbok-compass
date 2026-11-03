import pytest

from app.contracts import ParentChunk, SourceSpan
from app.corpus.structure_map import LogicalSection

from app.corpus import SourcePage, chunk_pages, segment_children, segment_parents


def parent(text: str) -> ParentChunk:
    return ParentChunk(parent_id="parent-1", corpus_version_id="v-1", section_path=("Data",), page_start=1, page_end=1, source_spans=[SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)], content_hash="a" * 64, token_count=len(text.split()), text=text)


def test_child_segmentation_is_deterministic_and_parent_linked() -> None:
    source = parent("one two three\nfour five six\nseven")
    first = segment_children([source], target_tokens=4, max_tokens=10)
    assert first == segment_children([source], target_tokens=4, max_tokens=10)
    assert [child.parent_id for child in first] == ["parent-1", "parent-1"]
    assert first[-1].text.endswith("seven")


def section(*spans: tuple[int, str]) -> LogicalSection:
    return LogicalSection(
        ("Data", "Governance"),
        tuple(SourceSpan(page=page, start_offset=0, end_offset=len(text), text=text) for page, text in spans),
    )


def test_parent_segmentation_preserves_cross_page_source_spans_and_is_repeatable() -> None:
    source = section((1, "one two\nthree four"), (2, "five six"))

    first = segment_parents([source], corpus_version_id="v1", target_tokens=4, max_tokens=6)
    second = segment_parents([source], corpus_version_id="v1", target_tokens=4, max_tokens=6)

    assert first == second
    assert first.ready_for_promotion
    assert [(parent.page_start, parent.page_end) for parent in first.parents] == [(1, 1), (2, 2)]
    assert [span.text for parent in first.parents for span in parent.source_spans] == ["one two", "three four", "five six"]


def test_parent_segmentation_keeps_short_sections_and_blocks_indivisible_rows() -> None:
    short = section((1, "short section"))
    oversized_row = section((2, "| " + "word " * 1_501 + "|"))

    result = segment_parents([short, oversized_row], corpus_version_id="v1", target_tokens=10, max_tokens=1_500)

    assert [parent.text for parent in result.parents] == ["short section"]
    assert not result.ready_for_promotion
    assert result.exceptions[0].page_start == result.exceptions[0].page_end == 2
    assert "table row" in result.exceptions[0].reason


def test_chunking_never_crosses_pages_and_preserves_provenance() -> None:
    pages = [
        SourcePage(page=2, section="Data governance", text="one two three four five"),
        SourcePage(page=3, section="Metadata", text="six seven"),
    ]

    chunks = chunk_pages(pages, corpus_version_id="v-abc", target_tokens=3, overlap_tokens=1)

    assert [(chunk.page, chunk.section, chunk.chunk_ordinal) for chunk in chunks] == [
        (2, "Data governance", 0),
        (2, "Data governance", 1),
        (3, "Metadata", 0),
    ]
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "six seven"]
    assert all(chunk.corpus_version_id == "v-abc" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("v-abc:") for chunk in chunks)


def test_chunking_normalizes_whitespace_and_is_repeatable() -> None:
    page = SourcePage(page=1, section="  Overview  ", text=" alpha\n\tbeta  gamma ")

    first = chunk_pages([page], corpus_version_id="v-1", target_tokens=2, overlap_tokens=0)
    second = chunk_pages([page], corpus_version_id="v-1", target_tokens=2, overlap_tokens=0)

    assert first == second
    assert [chunk.text for chunk in first] == ["alpha beta", "gamma"]
    assert first[0].section == "Overview"
    assert len(first[0].content_hash) == 64


def test_empty_pages_are_ignored() -> None:
    chunks = chunk_pages(
        [SourcePage(page=1, section="Empty", text="\n  "), SourcePage(page=2, section="Body", text="kept")],
        corpus_version_id="v-1",
    )

    assert [chunk.page for chunk in chunks] == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_tokens": 0}, "target_tokens"),
        ({"target_tokens": 4, "overlap_tokens": 4}, "overlap_tokens"),
        ({"target_tokens": 4, "overlap_tokens": -1}, "overlap_tokens"),
        ({"corpus_version_id": " v-1"}, "corpus_version_id"),
    ],
)
def test_chunking_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    defaults: dict[str, object] = {"corpus_version_id": "v-1"}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        chunk_pages([SourcePage(page=1, section="Body", text="text")], **defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "page",
    [
        SourcePage(page=0, section="Body", text="text"),
        SourcePage(page=1, section=" ", text="text"),
    ],
)
def test_chunking_rejects_invalid_page_provenance(page: SourcePage) -> None:
    with pytest.raises(ValueError):
        chunk_pages([page], corpus_version_id="v-1")
