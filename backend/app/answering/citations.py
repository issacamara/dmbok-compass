"""Deterministic validation and rendering for grounded answer evidence.

Provider output is untrusted. This module verifies membership and exact source
spans against request-scoped retrieved records, then reconstructs public
citation metadata from those records. It deliberately does not claim semantic
entailment; that remains an evaluator/human-review concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from app.contracts import Citation, CitationEvidence, RetrievedPassage
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
) -> list[Citation]:
    """Validate every provider citation and return server-owned metadata.

    Validation is all-or-nothing: a valid citation must not mask an unknown,
    duplicated, mixed, or altered citation in the same response.
    """

    by_id = _passages_by_id(passages)
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
            )
        )
    return citations


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
