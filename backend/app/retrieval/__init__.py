"""Active-corpus retrieval and evidence services."""

from .evidence import (
    EvidenceBundle,
    EvidenceOutcome,
    MAX_EVIDENCE_PASSAGES,
    PARTIAL_EVIDENCE_THRESHOLD,
    RetrievalCase,
    STRONG_EVIDENCE_THRESHOLD,
    classify_evidence,
    retrieval_success_metric,
)
from .search import FirestorePassageRetriever
from .lexical import LexicalCandidate, LexicalIndex

__all__ = [
    "EvidenceBundle",
    "EvidenceOutcome",
    "FirestorePassageRetriever",
    "MAX_EVIDENCE_PASSAGES",
    "PARTIAL_EVIDENCE_THRESHOLD",
    "RetrievalCase",
    "STRONG_EVIDENCE_THRESHOLD",
    "classify_evidence",
    "retrieval_success_metric",
    "LexicalCandidate",
    "LexicalIndex",
]
