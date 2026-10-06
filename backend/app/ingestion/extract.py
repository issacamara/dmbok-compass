"""Extract searchable PDF text while retaining page and section provenance.

This module is deliberately a spike, not the corpus ingestion job. It produces
an inspectable report that can be reviewed before chunking or embedding is
implemented.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


_CHAPTER = re.compile(r"\b(?:CHAPTER|Chapter)\s+\d+(?:\s*[-:.]\s*|\s+)(.+)")
_NUMBERED_SECTION = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class PageProvenance:
    """Text and the source location that may be cited for one PDF page."""

    page: int
    section: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """Stable, reviewable output of one PDF extraction run."""

    source_uri: str
    source_sha256: str
    parser: str
    page_count: int
    pages: tuple[PageProvenance, ...]
    failures: tuple[str, ...]

    @property
    def section_recovery_rate(self) -> float:
        """Return the proportion of non-empty pages with a recovered section."""

        non_empty = [page for page in self.pages if page.text]
        if not non_empty:
            return 0.0
        return sum(page.section is not None for page in non_empty) / len(non_empty)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pages"] = [asdict(page) for page in self.pages]
        result["section_recovery_rate"] = self.section_recovery_rate
        return result

    def to_json(self) -> str:
        """Serialize the report with stable ordering for reproducibility."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _normalise_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _section_heading(line: str) -> str | None:
    chapter = _CHAPTER.search(line)
    if chapter:
        return f"Chapter {chapter.group(1).strip()}"
    numbered = _NUMBERED_SECTION.match(line)
    if numbered:
        return f"{numbered.group(1)} {numbered.group(2).strip()}"
    return None


def _recover_section(text: str, previous: str | None) -> str | None:
    """Recover the last chapter/numbered heading visible on a page."""

    section = previous
    for line in text.splitlines():
        heading = _section_heading(line)
        if heading:
            section = heading
    return section


def extract_pdf(pdf_path: str | Path, *, source_uri: str | None = None) -> ExtractionReport:
    """Extract page text and provenance from a searchable PDF.

    Pages are one-based, matching the citation contract. Section headings are
    recovered from chapter headings and numbered headings; a heading carries
    forward until another heading is encountered. Pages without text or a
    recoverable section are reported as failures rather than silently omitted.
    """

    path = Path(pdf_path)
    payload = path.read_bytes()
    reader = PdfReader(path)
    previous_section: str | None = None
    pages: list[PageProvenance] = []
    failures: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _normalise_text(raw_text)
        section = _recover_section(raw_text, previous_section)
        if text and section:
            previous_section = section
        if not text:
            failures.append(f"page {page_number}: no searchable text extracted")
        elif section is None:
            failures.append(f"page {page_number}: no chapter or numbered section heading recovered")
        pages.append(PageProvenance(page=page_number, section=section, text=text))

    parser_version = getattr(__import__("pypdf"), "__version__", "unknown")
    return ExtractionReport(
        source_uri=source_uri or path.name,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        parser=f"pypdf {parser_version}",
        page_count=len(reader.pages),
        pages=tuple(pages),
        failures=tuple(failures),
    )
