import pytest

from app.contracts import StructureMap, StructureMapEntry
from app.corpus.structure_map import sections_from_pages


SOURCE = "a" * 64


def test_sections_preserve_cross_page_coverage_and_offsets() -> None:
    structure_map = StructureMap(
        map_id="map", source_sha256=SOURCE, content_hash="b" * 64,
        entries=[
            StructureMapEntry(section_path=("H1", "H2"), page_start=1, page_end=2),
            StructureMapEntry(section_path=("H1", "H3"), page_start=3, page_end=3),
        ],
    )

    sections = sections_from_pages(structure_map, source_sha256=SOURCE, pages=[(1, "alpha"), (2, "beta"), (3, "gamma")])

    assert [section.section_path for section in sections] == [("H1", "H2"), ("H1", "H3")]
    assert sections[0].text == "alpha\nbeta"
    assert [(span.page, span.start_offset, span.end_offset) for span in sections[0].source_spans] == [(1, 0, 5), (2, 0, 4)]
    assert sections[0].page_start == 1
    assert sections[0].page_end == 2


def test_unclassified_path_never_reassigns_text() -> None:
    structure_map = StructureMap(
        map_id="map", source_sha256=SOURCE, content_hash="b" * 64,
        entries=[
            StructureMapEntry(section_path=("H1",), page_start=1, page_end=1),
            StructureMapEntry(section_path=("Unclassified",), page_start=2, page_end=2),
        ],
    )

    sections = sections_from_pages(structure_map, source_sha256=SOURCE, pages=[(1, "known"), (2, "uncertain")])

    assert sections[1].section_path == ("Unclassified",)
    assert sections[1].text == "uncertain"


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([StructureMapEntry(section_path=("H1",), page_start=2, page_end=2)], "contiguously"),
        ([StructureMapEntry(section_path=("H1",), page_start=1, page_end=1)], "every source page"),
    ],
)
def test_rejects_malformed_map_coverage(entries: list[StructureMapEntry], message: str) -> None:
    structure_map = StructureMap(map_id="map", source_sha256=SOURCE, content_hash="b" * 64, entries=entries)

    with pytest.raises(ValueError, match=message):
        sections_from_pages(structure_map, source_sha256=SOURCE, pages=[(1, "one"), (2, "two")])


def test_rejects_source_mismatch_and_non_contiguous_pages() -> None:
    structure_map = StructureMap(
        map_id="map", source_sha256=SOURCE, content_hash="b" * 64,
        entries=[StructureMapEntry(section_path=("H1",), page_start=1, page_end=1)],
    )

    with pytest.raises(ValueError, match="source SHA-256"):
        sections_from_pages(structure_map, source_sha256="c" * 64, pages=[(1, "one")])
    with pytest.raises(ValueError, match="ordered"):
        sections_from_pages(structure_map, source_sha256=SOURCE, pages=[(2, "one")])
