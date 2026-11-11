"""Deterministic validation and rendering for grounded answer evidence.

Provider output is untrusted. This module verifies membership and exact source
spans against request-scoped retrieved records, then reconstructs public
citation metadata from those records. It deliberately does not claim semantic
entailment; that remains an evaluator/human-review concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

from app.contracts import (
    ChildDocumentChunk,
    Citation,
    CitationEvidence,
    GenerationContext,
    ParentChunk,
    RetrievedPassage,
    SourceSpan,
)
from model.protocol import GenerationCandidate, GenerationError, GenerationErrorCode


class CitationValidationError(GenerationError):
    """Typed safe failure for an invalid candidate/evidence association."""

    def __init__(self, message: str) -> None:
        super().__init__(GenerationErrorCode.INVALID_OUTPUT, message)


@dataclass(frozen=True)
class ValidatedClaim:
    """A claim whose complete evidence association has been checked."""

    claim_id: str
    text: str
    evidence: tuple[CitationEvidence, ...]


def _passages_by_id(passages: Iterable[RetrievedPassage]) -> dict[str, RetrievedPassage]:
    indexed: dict[str, RetrievedPassage] = {}
    for passage in passages:
        if passage.chunk_id in indexed:
            raise CitationValidationError(
                f"retrieved evidence contains duplicate chunk ID: {passage.chunk_id}"
            )
        indexed[passage.chunk_id] = passage
    return indexed


def _validate_span(
    *,
    citation_id: str,
    page: int,
    section: str,
    excerpt: str,
    passage: RetrievedPassage,
) -> None:
    if page != passage.page or section != passage.section:
        raise CitationValidationError(
            f"citation {citation_id} has altered page or section provenance"
        )
    if not excerpt or excerpt not in passage.excerpt:
        raise CitationValidationError(
            f"citation {citation_id} does not identify an exact retrieved source span"
        )


def validated_citations(
    candidate: GenerationCandidate,
    passages: tuple[RetrievedPassage, ...],
    *,
    child_sources: Sequence[ChildDocumentChunk] | None = None,
    source_parents: Sequence[ParentChunk] = (),
    generation_context: GenerationContext | None = None,
    corpus_version_id: str | None = None,
) -> list[Citation]:
    """Validate every provider citation and return server-owned metadata.

    Validation is all-or-nothing: a valid citation must not mask an unknown,
    duplicated, mixed, or altered citation in the same response.
    """

    by_id = _passages_by_id(passages)
    if child_sources is not None:
        return _validated_child_citations(
            candidate,
            by_id,
            child_sources,
            source_parents,
            generation_context,
            corpus_version_id=corpus_version_id,
        )
    citations: list[Citation] = []
    seen: set[str] = set()
    for citation in candidate.citations:
        if citation.citation_id in seen:
            raise CitationValidationError(
                f"candidate contains duplicate citation ID: {citation.citation_id}"
            )
        passage = by_id.get(citation.citation_id)
        if passage is None:
            raise CitationValidationError(
                f"candidate cites evidence outside the retrieved set: {citation.citation_id}"
            )
        _validate_span(
            citation_id=citation.citation_id,
            page=citation.page,
            section=citation.section,
            excerpt=citation.excerpt,
            passage=passage,
        )
        seen.add(citation.citation_id)
        citations.append(
            Citation(
                citation_id=passage.chunk_id,
                page=passage.page,
                section=passage.section,
                excerpt=citation.excerpt[:500],
            )
        )
    return citations


def _validated_child_citations(
    candidate: GenerationCandidate,
    passages: Mapping[str, RetrievedPassage],
    children: Sequence[ChildDocumentChunk],
    parents: Sequence[ParentChunk],
    generation_context: GenerationContext | None,
    *,
    corpus_version_id: str | None,
) -> list[Citation]:
    if not corpus_version_id:
        raise CitationValidationError("citation validation requires an active corpus version")
    child_by_id = {child.chunk_id: child for child in children}
    parent_by_id = {parent.parent_id: parent for parent in parents}
    if len(child_by_id) != len(children) or len(parent_by_id) != len(parents):
        raise CitationValidationError("retrieved source provenance contains duplicate IDs")
    supplied_child_ids: set[str] | None = None
    supplied_parent_ids: set[str] | None = None
    if generation_context is not None:
        if generation_context.corpus_version_id != corpus_version_id:
            raise CitationValidationError("generation context belongs to another corpus")
        supplied_parent_ids = {parent.parent_id for parent in generation_context.parents}
        supplied_child_ids = {
            child_id
            for parent in generation_context.parents
            for child_id in parent.backing_child_ids
        }
        if not supplied_parent_ids or not supplied_child_ids:
            raise CitationValidationError("generation context has no source allowlist")
        for context_parent in generation_context.parents:
            parent = parent_by_id.get(context_parent.parent_id)
            if parent is None:
                raise CitationValidationError(
                    f"generation context parent is unavailable: {context_parent.parent_id}"
                )
            expected_spans = tuple(
                (span.span_id, span.block_id, span.page, span.printed_page_label,
                 span.start_offset, span.end_offset, span.text)
                for span in parent.source_spans
            )
            supplied_spans = tuple(
                (span.span_id, span.block_id, span.physical_page, span.printed_page_label,
                 span.start_offset, span.end_offset, span.raw_text)
                for span in context_parent.spans
            )
            actual_child_ids = {
                child.chunk_id for child in children if child.parent_id == parent.parent_id
            }
            if (
                context_parent.embedding_text != parent.text
                or supplied_spans != expected_spans
                or not set(context_parent.backing_child_ids).issubset(actual_child_ids)
            ):
                raise CitationValidationError(
                    f"generation context does not match stored parent provenance: {parent.parent_id}"
                )

    citations: list[Citation] = []
    seen: set[str] = set()
    for citation in candidate.citations:
        if citation.citation_id in seen:
            raise CitationValidationError(
                f"candidate contains duplicate citation ID: {citation.citation_id}"
            )
        child = child_by_id.get(citation.citation_id)
        if child is None or child.chunk_id not in passages:
            raise CitationValidationError(
                f"candidate cites evidence outside the retrieved child set: {citation.citation_id}"
            )
        if supplied_child_ids is not None and child.chunk_id not in supplied_child_ids:
            raise CitationValidationError(
                f"candidate cites a child outside supplied generation context: {child.chunk_id}"
            )
        passage = passages[child.chunk_id]
        if (
            passage.excerpt != child.text
            or passage.page != child.page_start
            or passage.section != " / ".join(child.section_path)[:500]
        ):
            raise CitationValidationError(
                f"retrieved child passage does not match its stored source: {child.chunk_id}"
            )
        parent = parent_by_id.get(child.parent_id)
        if parent is None:
            raise CitationValidationError(
                f"retrieved child has no request-local parent: {child.chunk_id}"
            )
        if supplied_parent_ids is not None and parent.parent_id not in supplied_parent_ids:
            raise CitationValidationError(
                f"retrieved child parent is outside supplied generation context: {child.chunk_id}"
            )
        if child.corpus_version_id != corpus_version_id or parent.corpus_version_id != corpus_version_id:
            raise CitationValidationError(
                f"citation evidence crosses corpus versions at chunk {child.chunk_id}"
            )
        _validate_child_parent_link(child, parent)
        source = _unique_matching_source_span(citation.excerpt, child)
        seen.add(citation.citation_id)
        citations.append(
            Citation(
                citation_id=child.chunk_id,
                page=source.page,
                section=" / ".join(parent.section_path)[:500],
                excerpt=citation.excerpt[:500],
            )
        )
    return citations


def _validate_child_parent_link(child: ChildDocumentChunk, parent: ParentChunk) -> None:
    if child.parent_id != parent.parent_id or child.section_path != parent.section_path:
        raise CitationValidationError(
            f"retrieved child has an invalid parent link: {child.chunk_id}"
        )
    if (
        (parent.source_text or parent.text) != "\n".join(span.text for span in parent.source_spans)
        or (child.source_text or child.text) != "\n".join(span.text for span in child.source_spans)
        or any(
            span.end_offset - span.start_offset != len(span.text)
            for span in (*parent.source_spans, *child.source_spans)
        )
    ):
        raise CitationValidationError(f"retrieved source text or offsets changed: {child.chunk_id}")
    if (
        sha256(parent.text.encode("utf-8")).hexdigest() != parent.content_hash
        or sha256(child.text.encode("utf-8")).hexdigest() != child.content_hash
    ):
        raise CitationValidationError(f"retrieved source content hash changed: {child.chunk_id}")
    for child_span in child.source_spans:
        linked = any(
            source.page == child_span.page
            and source.block_id == child_span.block_id
            and source.start_offset <= child_span.start_offset
            and child_span.end_offset <= source.end_offset
            and source.text[
                child_span.start_offset - source.start_offset:
                child_span.end_offset - source.start_offset
            ] == child_span.text
            for source in parent.source_spans
        )
        if not linked:
            raise CitationValidationError(
                f"retrieved child source span is not linked to its parent: {child.chunk_id}"
            )


def _unique_matching_source_span(excerpt: str, child: ChildDocumentChunk) -> SourceSpan:
    if not excerpt:
        raise CitationValidationError(
            f"citation {child.chunk_id} has an empty source excerpt"
        )
    matches = [span for span in child.source_spans if excerpt in span.text]
    if sum(span.text.count(excerpt) for span in matches) != 1:
        matches = []
    if len(matches) != 1:
        raise CitationValidationError(
            f"citation {child.chunk_id} does not identify one exact source span"
        )
    return matches[0]


def validate_citation_evidence(
    evidence: Sequence[CitationEvidence],
    passages: tuple[RetrievedPassage, ...],
    claims: Mapping[str, str] | None = None,
    *,
    corpus_version_id: str | None = None,
) -> tuple[ValidatedClaim, ...]:
    """Validate frozen claim/evidence records against retrieved source spans.

    ``claims`` is a claim-id-to-text mapping from the candidate boundary. If
    supplied, every claim must be linked and every evidence record must point
    to a known claim. Optional corpus metadata is checked when retrieved
    records expose it, keeping older passage fixtures compatible.
    """

    by_id = _passages_by_id(passages)
    known_claims = dict(claims or {})
    linked: dict[str, list[CitationEvidence]] = {claim_id: [] for claim_id in known_claims}
    seen_citations: set[str] = set()
    for item in evidence:
        if item.citation_id in seen_citations:
            raise CitationValidationError(
                f"duplicate citation evidence ID: {item.citation_id}"
            )
        passage = by_id.get(item.chunk_id)
        if passage is None:
            raise CitationValidationError(
                f"citation evidence references an unknown chunk: {item.chunk_id}"
            )
        if known_claims and item.claim_id not in known_claims:
            raise CitationValidationError(
                f"citation evidence is not linked to a known claim: {item.claim_id}"
            )
        if corpus_version_id is not None:
            passage_corpus = getattr(passage, "corpus_version_id", corpus_version_id)
            if passage_corpus != corpus_version_id:
                raise CitationValidationError(
                    f"citation evidence crosses corpus versions at chunk {item.chunk_id}"
                )
        _validate_span(
            citation_id=item.citation_id,
            page=item.page,
            section=item.section,
            excerpt=item.excerpt,
            passage=passage,
        )
        seen_citations.add(item.citation_id)
        if known_claims:
            linked[item.claim_id].append(item)

    if known_claims:
        missing = [claim_id for claim_id, items in linked.items() if not items]
        if missing:
            raise CitationValidationError(
                "claims without citation evidence: " + ", ".join(sorted(missing))
            )
        return tuple(
            ValidatedClaim(claim_id=claim_id, text=text, evidence=tuple(linked[claim_id]))
            for claim_id, text in known_claims.items()
        )
    return ()


def render_validated_claims(
    claims: Sequence[ValidatedClaim], *, connective_text: str = " "
) -> str:
    """Render only validated claim text joined by fixed connective text."""

    if not connective_text or "\n" in connective_text:
        raise ValueError("connective_text must be a fixed single-line separator")
    if not claims:
        raise CitationValidationError("cannot render an answer without validated claims")
    return connective_text.join(claim.text for claim in claims)
