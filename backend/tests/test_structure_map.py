import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.corpus.structure_map import HeadingAnchor, SourceIdentity, propose_structure_map


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
