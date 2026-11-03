"""Build deterministic, review-only corpus structure-map proposals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.contracts import SourceSpan, StructureMap, StructureMapEntry


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNCLASSIFIED = ("Unclassified",)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Immutable source identity required for a proposal artifact."""

    uri: str
    generation: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri or self.uri.strip() != self.uri:
            raise ValueError("source URI must be a non-empty normalized value")
        if self.generation < 1:
            raise ValueError("source generation must be positive")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("source SHA-256 must be a lowercase digest")


@dataclass(frozen=True, slots=True)
class HeadingAnchor:
    """One table-of-contents or printed-heading assertion for a source page."""

    page: int
    section_path: tuple[str, ...]
    heading: str

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("heading page must be positive")
        if not 1 <= len(self.section_path) <= 4 or any(not value.strip() for value in self.section_path):
            raise ValueError("heading section path must contain one to four non-empty levels")
        if not self.heading.strip():
            raise ValueError("heading text must be non-empty")


@dataclass(frozen=True, slots=True)
class UnresolvedDecision:
    page: int
    reason: str


@dataclass(frozen=True, slots=True)
class StructureMapProposal:
    """Candidate map with review metadata; it is never an approval record."""

    schema_version: str
    source: SourceIdentity
    structure_map: StructureMap
    unresolved: tuple[UnresolvedDecision, ...]

    @property
    def source_coverage(self) -> float:
        entries = self.structure_map.entries
        total = sum(entry.page_end - entry.page_start + 1 for entry in entries)
        classified = sum(
            entry.page_end - entry.page_start + 1
            for entry in entries
            if entry.section_path != _UNCLASSIFIED
        )
        return classified / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_status": "proposed",
            "source": {"uri": self.source.uri, "generation": self.source.generation, "sha256": self.source.sha256},
            "structure_map": self.structure_map.model_dump(mode="json"),
            "source_coverage": self.source_coverage,
            "unresolved": [{"page": item.page, "reason": item.reason} for item in self.unresolved],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class LogicalSection:
    """One ordered, source-linked section recovered from mapped PDF pages."""

    section_path: tuple[str, ...]
    source_spans: tuple[SourceSpan, ...]

    @property
    def page_start(self) -> int:
        return self.source_spans[0].page

    @property
    def page_end(self) -> int:
        return self.source_spans[-1].page

    @property
    def text(self) -> str:
        return "\n".join(span.text for span in self.source_spans)


def sections_from_pages(
    structure_map: StructureMap,
    *,
    source_sha256: str,
    pages: Iterable[tuple[int, str]],
) -> tuple[LogicalSection, ...]:
    """Map ordered one-based pages to contiguous logical sections.

    The map is deliberately authoritative: malformed coverage or a source
    mismatch is rejected instead of silently assigning text to a neighbour.
    """

    if structure_map.source_sha256 != source_sha256:
        raise ValueError("structure map source SHA-256 does not match extracted source")
    page_text = tuple(pages)
    expected_pages = tuple(range(1, len(page_text) + 1))
    if tuple(page for page, _ in page_text) != expected_pages:
        raise ValueError("source pages must be ordered, contiguous, and one-based")
    paths = _paths_by_page(structure_map.entries, len(page_text))
    sections: list[LogicalSection] = []
    current_path: tuple[str, ...] | None = None
    spans: list[SourceSpan] = []
    for (page, text), path in zip(page_text, paths, strict=True):
        if path != current_path:
            if spans:
                sections.append(LogicalSection(current_path or _UNCLASSIFIED, tuple(spans)))
            current_path, spans = path, []
        if text:
            spans.append(SourceSpan(page=page, start_offset=0, end_offset=len(text), text=text))
    if spans:
        sections.append(LogicalSection(current_path or _UNCLASSIFIED, tuple(spans)))
    return tuple(sections)


def propose_structure_map(
    source: SourceIdentity,
    *,
    page_count: int,
    toc: Iterable[HeadingAnchor],
    printed_headings: Iterable[HeadingAnchor],
) -> StructureMapProposal:
    """Reconcile heading evidence into a candidate map with explicit uncertainty."""

    if page_count < 1:
        raise ValueError("page_count must be positive")
    toc_by_page = _group(toc, page_count, "table-of-contents")
    printed_by_page = _group(printed_headings, page_count, "printed-heading")
    paths: list[tuple[str, ...]] = []
    unresolved: list[UnresolvedDecision] = []
    current = _UNCLASSIFIED
    for page in range(1, page_count + 1):
        toc_paths = toc_by_page.get(page, set())
        printed_paths = printed_by_page.get(page, set())
        if toc_paths or printed_paths:
            if len(toc_paths) == len(printed_paths) == 1 and toc_paths == printed_paths:
                current = next(iter(toc_paths))
            else:
                current = _UNCLASSIFIED
                unresolved.append(UnresolvedDecision(page, _mismatch_reason(toc_paths, printed_paths)))
        paths.append(current)
    entries = _entries(paths)
    material = [{"section_path": entry.section_path, "page_start": entry.page_start, "page_end": entry.page_end} for entry in entries]
    content_hash = _sha256(_canonical_json(material))
    structure_map = StructureMap(
        map_id=f"proposal-{source.sha256[:12]}-{content_hash[:12]}",
        source_sha256=source.sha256,
        content_hash=content_hash,
        entries=entries,
    )
    return StructureMapProposal("structure-map-proposal-v1", source, structure_map, tuple(unresolved))


def _group(anchors: Iterable[HeadingAnchor], page_count: int, label: str) -> dict[int, set[tuple[str, ...]]]:
    grouped: dict[int, set[tuple[str, ...]]] = {}
    for anchor in anchors:
        if anchor.page > page_count:
            raise ValueError(f"{label} heading page exceeds source page count")
        grouped.setdefault(anchor.page, set()).add(anchor.section_path)
    return grouped


def _entries(paths: list[tuple[str, ...]]) -> list[StructureMapEntry]:
    entries: list[StructureMapEntry] = []
    start = 1
    current = paths[0]
    for page, path in enumerate(paths[1:], start=2):
        if path != current:
            entries.append(StructureMapEntry(section_path=current, page_start=start, page_end=page - 1))
            start, current = page, path
    entries.append(StructureMapEntry(section_path=current, page_start=start, page_end=len(paths)))
    return entries


def _paths_by_page(entries: Iterable[StructureMapEntry], page_count: int) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    next_page = 1
    for entry in entries:
        if entry.page_start != next_page or entry.page_end > page_count:
            raise ValueError("structure map entries must cover source pages contiguously")
        paths.extend([entry.section_path] * (entry.page_end - entry.page_start + 1))
        next_page = entry.page_end + 1
    if next_page != page_count + 1:
        raise ValueError("structure map entries must cover every source page")
    return tuple(paths)


def _mismatch_reason(toc_paths: set[tuple[str, ...]], printed_paths: set[tuple[str, ...]]) -> str:
    if not toc_paths:
        return "missing table-of-contents anchor"
    if not printed_paths:
        return "missing printed-heading anchor"
    if len(toc_paths) > 1 or len(printed_paths) > 1:
        return "ambiguous heading anchors"
    return "table-of-contents and printed-heading paths disagree"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
