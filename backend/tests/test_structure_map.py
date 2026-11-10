import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.corpus.structure_map import (
    HeadingAnchor,
    SourceIdentity,
    correct_block_structure_map,
    propose_block_structure_map,
    propose_structure_map,
    validate_block_structure_map,
    sections_from_parsing_artifact,
)
from app.contracts import StructureMap, StructureMapEntry
from app.contracts.v29 import ParsedBlock, ParsingArtifact


SOURCE = SourceIdentity("gs://bucket/dmbok.pdf#12", 12, "a" * 64)


def anchor(page: int, path: tuple[str, ...]) -> HeadingAnchor:
    return HeadingAnchor(page, path, path[-1])


def test_proposal_is_deterministic_and_tracks_source_identity() -> None:
    proposal = propose_structure_map(
        SOURCE,
        page_count=4,
        toc=[anchor(1, ("Chapter 1",)), anchor(3, ("Chapter 2",))],
        printed_headings=[anchor(1, ("Chapter 1",)), anchor(3, ("Chapter 2",))],
    )
    repeated = propose_structure_map(
        SOURCE,
        page_count=4,
        toc=[anchor(1, ("Chapter 1",)), anchor(3, ("Chapter 2",))],
        printed_headings=[anchor(1, ("Chapter 1",)), anchor(3, ("Chapter 2",))],
    )
    assert proposal.to_json() == repeated.to_json()
    assert proposal.structure_map.entries[0].section_path == ("Chapter 1",)
    assert proposal.to_dict()["review_status"] == "proposed"
    assert proposal.to_dict()["source"]["generation"] == 12


def test_missing_or_ambiguous_heading_becomes_unclassified() -> None:
    proposal = propose_structure_map(
        SOURCE,
        page_count=3,
        toc=[anchor(1, ("Chapter 1",)), anchor(2, ("Chapter 2",)), anchor(2, ("Appendix",))],
        printed_headings=[anchor(1, ("Chapter 1",))],
    )
    assert [(entry.section_path, entry.page_start, entry.page_end) for entry in proposal.structure_map.entries] == [
        (("Chapter 1",), 1, 1),
        (("Unclassified",), 2, 3),
    ]
    assert [item.reason for item in proposal.unresolved] == ["missing printed-heading anchor"]

    ambiguous = propose_structure_map(
        SOURCE,
        page_count=1,
        toc=[anchor(1, ("Chapter 1",)), anchor(1, ("Appendix",))],
        printed_headings=[anchor(1, ("Chapter 1",)), anchor(1, ("Appendix",))],
    )
    assert ambiguous.unresolved[0].reason == "ambiguous heading anchors"


def test_page_anchor_changes_and_source_mismatches_are_detected() -> None:
    base = propose_structure_map(SOURCE, page_count=3, toc=[anchor(1, ("One",))], printed_headings=[anchor(1, ("One",))])
    changed = propose_structure_map(SOURCE, page_count=3, toc=[anchor(2, ("One",))], printed_headings=[anchor(2, ("One",))])
    assert base.structure_map.content_hash != changed.structure_map.content_hash
    with pytest.raises(ValueError, match="source SHA-256"):
        SourceIdentity("gs://bucket/dmbok.pdf#12", 12, "not-a-digest")


def test_block_proposal_is_complete_page_accurate_and_parse_bound() -> None:
    parsing = ParsingArtifact(
        source_generation="12", source_sha256=SOURCE.sha256, output_sha256="b" * 64,
        blocks=(
            ParsedBlock(block_id="b0", physical_page=4, reading_order=0, kind="text", raw_text="# Chapter 1"),
            ParsedBlock(block_id="b1", physical_page=4, reading_order=1, kind="text", raw_text="Overview"),
            ParsedBlock(block_id="b2", physical_page=5, reading_order=2, kind="text", raw_text="## 1.1 Roles"),
        ),
    )

    proposal = propose_block_structure_map(SOURCE, parsing)

    assert proposal.parsing_output_sha256 == parsing.output_sha256
    assert [(entry.section_path, entry.page_start, entry.page_end, entry.block_ids) for entry in proposal.entries] == [
        (("Chapter 1",), 4, 4, ("b0", "b1")),
        (("Chapter 1", "1.1 Roles"), 5, 5, ("b2",)),
    ]
    validate_block_structure_map(parsing, proposal)


def test_missing_duplicate_and_unclassified_blocks_cannot_form_an_approvable_map() -> None:
    parsing = ParsingArtifact(
        source_generation="12", source_sha256=SOURCE.sha256, output_sha256="b" * 64,
        blocks=(
            ParsedBlock(block_id="b0", physical_page=1, reading_order=0, kind="text", raw_text="opening"),
            ParsedBlock(block_id="b1", physical_page=1, reading_order=1, kind="text", raw_text="# Chapter 1"),
        ),
    )
    proposal = propose_block_structure_map(SOURCE, parsing)
    with pytest.raises(ValueError, match="cover every parse block"):
        correct_block_structure_map(SOURCE, parsing, [
            StructureMapEntry(section_path=("Chapter 1",), page_start=1, page_end=1, block_ids=("b1",)),
        ])
    with pytest.raises(ValueError, match="cover every parse block"):
        correct_block_structure_map(SOURCE, parsing, [
            StructureMapEntry(section_path=("Chapter 1",), page_start=1, page_end=1, block_ids=("b0", "b1")),
            StructureMapEntry(section_path=("Chapter 1", "duplicate"), page_start=1, page_end=1, block_ids=("b1",)),
        ])
    with pytest.raises(ValueError, match="does not cover"):
        validate_block_structure_map(parsing, proposal)


def test_script_binds_actual_source_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"immutable source")
    headings = tmp_path / "headings.json"
    headings.write_text(json.dumps({"page_count": 1, "toc": [{"page": 1, "section_path": ["One"], "heading": "One"}], "printed_headings": [{"page": 1, "section_path": ["One"], "heading": "One"}]}))
    output = tmp_path / "proposal.json"
    result = subprocess.run(
        [
            sys.executable, "scripts/propose_structure_map.py", str(source), str(headings),
            "--source-uri", "gs://bucket/dmbok.pdf#12", "--generation", "12",
            "--source-sha256", "fc17afe4af56fca9d2943b7901e7517611b37a36db7a7775b3e341e7d20a6ba0",
            "--output", str(output),
        ],
        cwd=Path(__file__).parents[2],
        env={"PYTHONPATH": "backend"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["source"]["sha256"] != "a" * 64
    mismatch = subprocess.run(
        [sys.executable, "scripts/propose_structure_map.py", str(source), str(headings), "--source-uri", "gs://bucket/dmbok.pdf#12", "--generation", "12", "--source-sha256", "a" * 64, "--output", str(output)],
        cwd=Path(__file__).parents[2], env={"PYTHONPATH": "backend"}, capture_output=True, text=True, check=False,
    )
    assert mismatch.returncode != 0
    assert "does not match" in mismatch.stderr


def test_reviewed_parse_sections_use_cumulative_page_offsets_and_exact_bounds():
    parsing = ParsingArtifact(
        source_generation="3", source_sha256="a" * 64, output_sha256="b" * 64,
        blocks=(
            ParsedBlock(block_id="b1", physical_page=4, reading_order=0, kind="text", raw_text="first"),
            ParsedBlock(block_id="b2", physical_page=4, reading_order=1, kind="text", raw_text="second"),
        ),
    )
    structure_map = StructureMap(
        map_id="reviewed", source_sha256=parsing.source_sha256,
        parsing_output_sha256=parsing.output_sha256, content_hash="c" * 64,
        entries=[StructureMapEntry(section_path=("Data",), page_start=4, page_end=4, block_ids=("b1", "b2"))],
    )

    section = sections_from_parsing_artifact(
        structure_map, parsing, source_generation=3, source_sha256=parsing.source_sha256
    )[0]
    assert [(span.start_offset, span.end_offset) for span in section.source_spans] == [(0, 5), (6, 12)]

    split_map = StructureMap(
        map_id="reviewed-split", source_sha256=parsing.source_sha256,
        parsing_output_sha256=parsing.output_sha256, content_hash="d" * 64,
        entries=[
            StructureMapEntry(section_path=("First",), page_start=4, page_end=4, block_ids=("b1",)),
            StructureMapEntry(section_path=("Second",), page_start=4, page_end=4, block_ids=("b2",)),
        ],
    )
    split_sections = sections_from_parsing_artifact(
        split_map, parsing, source_generation=3, source_sha256=parsing.source_sha256
    )
    assert [section.source_spans[0].start_offset for section in split_sections] == [0, 6]


def test_reviewed_parse_sections_reject_inaccurately_bounded_blocks():
    parsing = ParsingArtifact(
        source_generation="3", source_sha256="a" * 64, output_sha256="b" * 64,
        blocks=(ParsedBlock(block_id="b1", physical_page=4, reading_order=0, kind="text", raw_text="first"),),
    )
    structure_map = StructureMap(
        map_id="reviewed", source_sha256=parsing.source_sha256,
        parsing_output_sha256=parsing.output_sha256, content_hash="c" * 64,
        entries=[StructureMapEntry(section_path=("Data",), page_start=3, page_end=5, block_ids=("b1",))],
    )

    with pytest.raises(ValueError, match="page bounds"):
        sections_from_parsing_artifact(
            structure_map, parsing, source_generation=3, source_sha256=parsing.source_sha256
        )
