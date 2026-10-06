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
from .search import (
    QUERY_TASK_TYPE,
    VertexQueryEmbedder,
    FirestorePassageRetriever,
    create_vertex_query_embedder,
)

__all__ = [
    "EvidenceBundle",
    "EvidenceOutcome",
    "FirestorePassageRetriever",
    "MAX_EVIDENCE_PASSAGES",
    "PARTIAL_EVIDENCE_THRESHOLD",
    "QUERY_TASK_TYPE",
    "RetrievalCase",
    "STRONG_EVIDENCE_THRESHOLD",
    "VertexQueryEmbedder",
    "classify_evidence",
    "create_vertex_query_embedder",
    "retrieval_success_metric",
]
