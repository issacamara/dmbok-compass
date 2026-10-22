from __future__ import annotations

import pytest

from app.answering.citations import (
    CitationValidationError,
    render_validated_claims,
    validate_citation_evidence,
    validated_citations,
)
from app.contracts import CitationEvidence, RetrievedPassage
from model.protocol import GenerationCandidate


def passage(chunk_id: str = "chunk-1") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=4,
        section="Governance",
        excerpt="The retention policy requires review every year.",
        relevance_score=0.9,
    )


def candidate(*, citation_id: str = "chunk-1", **overrides: object) -> GenerationCandidate:
    citation = {
        "citation_id": citation_id,
        "page": 4,
        "section": "Governance",
        "excerpt": "retention policy requires review",
        **overrides,
    }
    return GenerationCandidate(
        outcome="answer",
        answer_text="The policy requires annual review.",
        citations=[citation],
    )


def evidence(*, claim_id: str = "claim-1", **overrides: object) -> CitationEvidence:
    values = {
        "citation_id": "citation-1",
        "claim_id": claim_id,
        "chunk_id": "chunk-1",
        "page": 4,
        "section": "Governance",
        "excerpt": "retention policy requires review",
    }
    values.update(overrides)
    return CitationEvidence(**values)


def test_valid_citation_reconstructs_server_owned_provenance() -> None:
    result = validated_citations(candidate(), (passage(),))

    assert result[0].model_dump() == {
        "citation_id": "chunk-1",
        "page": 4,
        "section": "Governance",
    }


@pytest.mark.parametrize(
    "bad_candidate",
    [
        candidate(citation_id="not-retrieved"),
        candidate(page=99),
        candidate(section="Fabricated"),
        candidate(excerpt="altered source text"),
    ],
)
def test_unknown_or_altered_provider_citations_fail_closed(
    bad_candidate: GenerationCandidate,
) -> None:
    with pytest.raises(CitationValidationError) as raised:
        validated_citations(bad_candidate, (passage(),))

    assert raised.value.code.value == "invalid_output"


def test_mixed_valid_and_invalid_citations_are_not_silently_filtered() -> None:
    mixed = GenerationCandidate(
        outcome="answer",
        answer_text="Mixed.",
        citations=[candidate().citations[0], candidate(citation_id="missing").citations[0]],
    )

    with pytest.raises(CitationValidationError):
        validated_citations(mixed, (passage(),))


def test_claim_evidence_requires_known_claims_and_exact_spans() -> None:
    claims = {"claim-1": "The policy requires annual review."}

    validated = validate_citation_evidence((evidence(),), (passage(),), claims)

    assert validated[0].claim_id == "claim-1"
    assert validated[0].evidence[0].chunk_id == "chunk-1"
    with pytest.raises(CitationValidationError, match="known claim"):
        validate_citation_evidence((evidence(claim_id="unlinked"),), (passage(),), claims)
    with pytest.raises(CitationValidationError, match="exact retrieved source span"):
        validate_citation_evidence(
            (evidence(excerpt="altered source text"),), (passage(),), claims
        )


def test_unlinked_claims_are_rejected_and_rendering_is_claim_only() -> None:
    with pytest.raises(CitationValidationError, match="without citation evidence"):
        validate_citation_evidence(
            (evidence(),), (passage(),), {"claim-1": "supported", "claim-2": "unsupported"}
        )

    claims = validate_citation_evidence(
        (evidence(),), (passage(),), {"claim-1": "supported factual claim"}
    )
    assert render_validated_claims(claims, connective_text="; ") == "supported factual claim"

