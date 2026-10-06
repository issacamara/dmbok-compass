"""Citation validation for request-scoped grounded answers."""

from __future__ import annotations

from app.contracts import Citation, RetrievedPassage
from model.protocol import GenerationCandidate


def validated_citations(
    candidate: GenerationCandidate,
    passages: tuple[RetrievedPassage, ...],
) -> list[Citation]:
    """Map model citation IDs to immutable provenance from retrieved chunks."""

    by_id = {passage.chunk_id: passage for passage in passages}
    citations: list[Citation] = []
    seen: set[str] = set()
    for citation in candidate.citations:
        passage = by_id.get(citation.citation_id)
        if passage is None or citation.citation_id in seen:
            continue
        seen.add(citation.citation_id)
        citations.append(
            Citation(
                citation_id=passage.chunk_id,
                page=passage.page,
                section=passage.section,
                excerpt=passage.excerpt,
            )
        )
    return citations
