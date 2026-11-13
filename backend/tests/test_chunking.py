import pytest

from app.contracts import ParentChunk, SourceSpan
from app.contracts import StructureMap, StructureMapEntry
from app.contracts.v29 import ParsedBlock, ParsingArtifact
from app.corpus.structure_map import LogicalSection, sections_from_parsing_artifact

from app.corpus import SourcePage, chunk_pages, segment_children, segment_parents
from app.corpus.tokenization import count_tokens


def parent(text: str) -> ParentChunk:
    return ParentChunk(parent_id="parent-1", corpus_version_id="v-1", section_path=("Data",), page_start=1, page_end=1, source_spans=[SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)], content_hash="a" * 64, token_count=len(text.split()), text=text)


def test_child_segmentation_is_deterministic_and_parent_linked() -> None:
    source = parent("one two three\nfour five six\nseven")
    first = segment_children([source], target_tokens=4, max_tokens=10)
    assert first == segment_children([source], target_tokens=4, max_tokens=10)
    assert [child.parent_id for child in first] == ["parent-1"] * len(first)
    assert all(child.token_count <= 10 for child in first)
    assert first[-1].text.endswith("seven")


def section(*spans: tuple[int, str]) -> LogicalSection:
    return LogicalSection(
        ("Data", "Governance"),
        tuple(SourceSpan(page=page, start_offset=0, end_offset=len(text), text=text) for page, text in spans),
    )


def test_parent_segmentation_preserves_cross_page_source_spans_and_is_repeatable() -> None:
    source = section((1, "one two\nthree four"), (2, "five six"))

    first = segment_parents([source], corpus_version_id="v1", target_tokens=8, max_tokens=12)
    second = segment_parents([source], corpus_version_id="v1", target_tokens=8, max_tokens=12)

    assert first == second
    assert first.ready_for_promotion
    assert all(parent.page_start == parent.page_end for parent in first.parents)
    assert [(parent.page_start, parent.page_end) for parent in first.parents] == [(1, 1), (1, 1), (2, 2)]
    assert [span.text for parent in first.parents for span in parent.source_spans] == ["one two", "three four", "five six"]


def test_parent_segmentation_keeps_short_sections_and_blocks_indivisible_rows() -> None:
    short = section((1, "short section"))
    oversized_row = section((2, "| " + "word " * 1_501 + "|"))

    result = segment_parents([short, oversized_row], corpus_version_id="v1", target_tokens=10, max_tokens=1_500)

    assert [parent.source_text for parent in result.parents] == ["short section"]
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


def test_long_prose_is_split_at_token_boundaries_with_source_offsets():
    text = " ".join(f"word{index}" for index in range(40))
    result = segment_parents(
        [LogicalSection(("Data",), (SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text),))],
        corpus_version_id="v1",
        target_tokens=10,
        max_tokens=20,
    )

    assert result.ready_for_promotion
    assert all(parent.token_count <= 20 for parent in result.parents)
    assert " ".join(span.text for parent in result.parents for span in parent.source_spans) == text


def test_children_repeat_markdown_table_header_without_claiming_source_span():
    text = "| Name | Value |\n| --- | --- |\n| first | one |\n| second | two |"
    children = segment_children([parent(text)], target_tokens=10, max_tokens=20)

    repeated = [
        child for child in children
        if "\n| Name | Value |\n| --- | --- |" in child.text
        and not any(span.text == "| Name | Value |" for span in child.source_spans)
    ]
    assert repeated
    assert all(child.token_count == count_tokens(child.text) for child in children)
    assert all(child.token_count <= 20 for child in children)


def test_parent_table_header_is_repeated_across_parent_boundaries():
    text = "| Name | Value |\n| --- | --- |\n" + "".join(
        f"| row-{index} | value-{index} |\n" for index in range(12)
    )
    source = LogicalSection(
        ("Data", "Table"),
        (SourceSpan(page=4, start_offset=0, end_offset=len(text), text=text),),
    )

    parents = segment_parents([source], corpus_version_id="v1", target_tokens=36, max_tokens=60)
    children = segment_children(parents.parents, target_tokens=24, max_tokens=60)

    assert parents.ready_for_promotion
    assert len(parents.parents) > 1
    assert all(parent.token_count <= 60 for parent in parents.parents)
    assert all(child.token_count <= 60 for child in children)
    for parent in parents.parents[1:]:
        assert parent.table_header == "| Name | Value |\n| --- | --- |"
        assert parent.text.startswith("Section: Data > Table\n| Name | Value |\n| --- | --- |\n")
    assert all(
        child.text.startswith("Section: Data > Table\n| Name | Value |\n| --- | --- |\n")
        for child in children
        if child.source_spans[0].text.startswith("| row-")
    )


def test_parent_and_child_keep_embedding_context_separate_from_exact_source_text():
    raw = "Reviewed text from the parser block."
    span = SourceSpan(
        page=7,
        start_offset=12,
        end_offset=12 + len(raw),
        text=raw,
        span_id="span-1",
        block_id="block-1",
        block_order=9,
        printed_page_label="vii",
    )
    parent_record = segment_parents(
        [LogicalSection(("Data", "Governance"), (span,))],
        corpus_version_id="v1",
        target_tokens=80,
        max_tokens=100,
    ).parents[0]
    child = segment_children([parent_record], target_tokens=80, max_tokens=100)[0]

    assert parent_record.source_text == raw
    assert parent_record.text == "Section: Data > Governance\n" + raw
    assert child.source_text == raw
    assert child.text == "Section: Data > Governance\n" + raw
    assert child.source_spans[0] == span


def test_llamaindex_nodes_link_children_to_parents_and_keep_provenance():
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, TextNode

    source = section((1, "A reviewed paragraph."))
    parents = segment_parents([source], corpus_version_id="v1").parents
    children = segment_children(parents)
    from app.corpus.chunking import build_llamaindex_nodes

    nodes = build_llamaindex_nodes(parents, children)
    by_id = {node.node_id: node for node in nodes}
    parent_node = by_id[parents[0].parent_id]
    child_node = by_id[children[0].chunk_id]

    assert isinstance(parent_node, TextNode)
    assert isinstance(child_node, TextNode)
    assert child_node.parent_node.node_id == parent_node.node_id
    assert [node.node_id for node in parent_node.child_nodes] == [child_node.node_id]
    assert child_node.metadata["source_text"] == children[0].source_text
    assert NodeRelationship.PARENT in child_node.relationships


def test_provenance_fields_survive_parent_and_child_contract_round_trip():
    span = SourceSpan(
        page=2,
        start_offset=4,
        end_offset=9,
        text="proof",
        span_id="span-2",
        block_id="block-2",
        block_order=3,
        printed_page_label="iv",
    )
    parent_record = segment_parents(
        [LogicalSection(("Evidence",), (span,))],
        corpus_version_id="v1",
    ).parents[0]
    child = segment_children([parent_record])[0]

    restored_parent = ParentChunk.model_validate_json(parent_record.model_dump_json())
    restored_child = type(child).model_validate_json(child.model_dump_json())

    assert restored_parent == parent_record
    assert restored_child == child
    assert restored_child.source_spans[0].block_id == "block-2"
    assert restored_child.source_spans[0].printed_page_label == "iv"


def test_csv_fallback_keeps_multiline_quoted_rows_atomic_and_repeats_header():
    raw_csv = 'Name,Notes\n"first\ncontinued",one\nsecond,two\nthird,three'
    parsing = ParsingArtifact(
        source_generation="3",
        source_sha256="a" * 64,
        output_sha256="b" * 64,
        blocks=(ParsedBlock(
            block_id="table-1",
            physical_page=5,
            reading_order=0,
            kind="table",
            raw_text=raw_csv,
            table_rows=(("Name", "Notes"), ("first\ncontinued", "one"), ("second", "two"), ("third", "three")),
        ),),
    )
    structure_map = StructureMap(
        map_id="reviewed-table",
        source_sha256=parsing.source_sha256,
        parsing_output_sha256=parsing.output_sha256,
        content_hash="c" * 64,
        entries=[StructureMapEntry(
            section_path=("Tables",), page_start=5, page_end=5, block_ids=("table-1",)
        )],
    )
    sections = sections_from_parsing_artifact(
        structure_map, parsing, source_generation=3, source_sha256=parsing.source_sha256
    )
    assert sections[0].source_spans[0].block_kind == "table"
    assert sections[0].source_spans[0].table_rows == parsing.blocks[0].table_rows

    parents = segment_parents(sections, corpus_version_id="v1", target_tokens=12, max_tokens=80)
    children = segment_children(parents.parents, target_tokens=8, max_tokens=80)

    assert parents.ready_for_promotion
    assert [span.text for parent_chunk in parents.parents for span in parent_chunk.source_spans] == [
        "Name,Notes", '"first\ncontinued",one', "second,two", "third,three"
    ]
    assert any(span.text == '"first\ncontinued",one' for child in children for span in child.source_spans)
    assert all(parent_chunk.token_count <= 80 for parent_chunk in parents.parents)
    assert all(child.token_count <= 80 for child in children)
    assert any(parent_chunk.table_header == "Name,Notes" for parent_chunk in parents.parents[1:])
    assert all(
        child.text.startswith("Section: Tables\nName,Notes\n")
        for child in children
        if child.source_spans[0].table_row_index not in (0, None)
    )


def test_parsed_list_items_remain_indivisible_in_parent_and_child_spans():
    raw_list = "- first item\n- second item with\n  continuation\n- third item"
    parsing = ParsingArtifact(
        source_generation="3",
        source_sha256="a" * 64,
        output_sha256="b" * 64,
        blocks=(ParsedBlock(
            block_id="list-1",
            physical_page=6,
            reading_order=0,
            kind="list",
            raw_text=raw_list,
            list_items=("first item", "second item with\n  continuation", "third item"),
        ),),
    )
    structure_map = StructureMap(
        map_id="reviewed-list",
        source_sha256=parsing.source_sha256,
        parsing_output_sha256=parsing.output_sha256,
        content_hash="d" * 64,
        entries=[StructureMapEntry(
            section_path=("Lists",), page_start=6, page_end=6, block_ids=("list-1",)
        )],
    )
    sections = sections_from_parsing_artifact(
        structure_map, parsing, source_generation=3, source_sha256=parsing.source_sha256
    )

    parents = segment_parents(sections, corpus_version_id="v1", target_tokens=8, max_tokens=80)
    children = segment_children(parents.parents, target_tokens=6, max_tokens=80)
    expected = ("- first item", "- second item with\n  continuation", "- third item")

    assert parents.ready_for_promotion
    assert tuple(span.text for parent_chunk in parents.parents for span in parent_chunk.source_spans) == expected
    assert tuple(span.list_item_index for child in children for span in child.source_spans) == (0, 1, 2)


def test_markdown_table_header_and_separator_stay_together_when_parent_splits():
    header_table = "| Name | Value |\n| --- | --- |\n| first | one |\n| second | two |"
    section = LogicalSection(
        ("Data",),
        (
            SourceSpan(
                page=3, start_offset=0, end_offset=len("preceding content"),
                text="preceding content", span_id="intro", block_id="intro",
                block_order=0, block_kind="text",
            ),
            SourceSpan(
                page=3, start_offset=0, end_offset=len(header_table), text=header_table,
                span_id="table", block_id="table", block_order=1, block_kind="table",
                table_rows=(("Name", "Value"), ("first", "one"), ("second", "two")),
            ),
        ),
    )

    parents = segment_parents([section], corpus_version_id="v1", target_tokens=10, max_tokens=80)
    children = segment_children(parents.parents, target_tokens=8, max_tokens=80)

    assert parents.ready_for_promotion
    assert len(parents.parents) > 1
    assert parents.parents[0].source_text == "preceding content"
    header_spans = [
        span
        for parent_chunk in parents.parents
        for span in parent_chunk.source_spans
        if span.table_row_index == 0
    ]
    assert len(header_spans) == 1
    assert header_spans[0].text == "| Name | Value |\n| --- | --- |"
    assert all(
        child.text.startswith("Section: Data\n| Name | Value |\n| --- | --- |\n")
        for child in children
        if child.source_spans[0].table_row_index not in (0, None)
    )


def test_markdown_table_rows_must_match_parsed_structured_rows():
    markdown = "| Name | Value |\n| --- | --- |\n| first | altered |"
    section = LogicalSection(
        ("Data",),
        (SourceSpan(
            page=3,
            start_offset=0,
            end_offset=len(markdown),
            text=markdown,
            span_id="table",
            block_id="table",
            block_kind="table",
            table_rows=(("Name", "Value"), ("first", "approved")),
        ),),
    )

    with pytest.raises(ValueError, match="Markdown table rows do not match"):
        segment_parents([section], corpus_version_id="v1")


def test_markdown_table_unescapes_brackets_before_matching_structured_rows():
    markdown = "| Name | Notes |\n| --- | --- |\n| first | cite \\[individuals\\] |"
    section = LogicalSection(
        ("Data",),
        (SourceSpan(
            page=3,
            start_offset=0,
            end_offset=len(markdown),
            text=markdown,
            span_id="table",
            block_id="table",
            block_kind="table",
            table_rows=(("Name", "Notes"), ("first", "cite [individuals]")),
        ),),
    )

    parents = segment_parents([section], corpus_version_id="v1")

    assert parents.ready_for_promotion
    assert parents.parents[0].source_spans[-1].text == "| first | cite \\[individuals\\] |"
