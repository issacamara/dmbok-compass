"""Query embedding and active-corpus nearest-neighbor retrieval."""

from __future__ import annotations

from typing import Any, Protocol

from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from app.contracts import RetrievedPassage
from app.corpus.embedding import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
VECTOR_FIELD = "embedding"
DISTANCE_RESULT_FIELD = "vector_distance"
DEFAULT_LIMIT = 5


class EmbeddingClient(Protocol):
    models: Any


class QueryEmbedder(Protocol):
    def embed(self, question: str) -> list[float]: ...


class VertexQueryEmbedder:
    """Generate a query vector compatible with corpus document vectors."""

    def __init__(self, client: EmbeddingClient, *, model: str = EMBEDDING_MODEL) -> None:
        self.client = client
        self.model = model

    def embed(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("question must be non-empty")
        response = self.client.models.embed_content(
            model=self.model,
            contents=question,
            config={
                "task_type": QUERY_TASK_TYPE,
                "output_dimensionality": EMBEDDING_DIMENSIONS,
            },
        )
        values = _embedding_values(response)
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Vertex returned {len(values)} dimensions; expected {EMBEDDING_DIMENSIONS}"
            )
        return values


def create_vertex_query_embedder(*, project: str, location: str) -> VertexQueryEmbedder:
    """Create the workload-identity Vertex client used for live questions."""
    from google import genai

    client = genai.Client(vertexai=True, project=project, location=location)
    return VertexQueryEmbedder(client)


class FirestorePassageRetriever:
    """Retrieve ranked passages from the currently active corpus version."""

    collection_id = "document_chunks"

    def __init__(self, client: Any, embedder: QueryEmbedder) -> None:
        self.chunks = client.collection(self.collection_id)
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        *,
        active_corpus_version_id: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[RetrievedPassage, ...]:
        """Return at most ``limit`` passages from the active corpus only."""
        if not active_corpus_version_id or not active_corpus_version_id.strip():
            raise ValueError("active_corpus_version_id must be non-empty")
        if limit < 1 or limit > DEFAULT_LIMIT:
            raise ValueError(f"limit must be between 1 and {DEFAULT_LIMIT}")

        query_vector = self.embedder.embed(question)
        active_query = self.chunks.where(
            filter=FieldFilter("corpus_version_id", "==", active_corpus_version_id)
        )
        nearest_query = active_query.find_nearest(
            vector_field=VECTOR_FIELD,
            query_vector=query_vector,
            distance_measure=DistanceMeasure.COSINE,
            limit=limit,
            distance_result_field=DISTANCE_RESULT_FIELD,
        )
        passages = [_passage_from_document(document) for document in nearest_query.stream()]
        passages.sort(key=_score_for_sorting, reverse=True)
        return tuple(passages[:limit])


def _passage_from_document(document: Any) -> RetrievedPassage:
    data = document.to_dict() if hasattr(document, "to_dict") else dict(document)
    chunk_id = data.get("chunk_id") or getattr(document, "id", "")
    excerpt = data.get("text") or data.get("excerpt")
    if not chunk_id or not excerpt:
        raise ValueError("retrieved document is missing chunk_id or text")

    distance = data.get(DISTANCE_RESULT_FIELD)
    score = data.get("relevance_score")
    if score is None and distance is not None:
        score = 1.0 - float(distance)
    return RetrievedPassage(
        chunk_id=chunk_id,
        page=data["page"],
        section=data["section"],
        excerpt=excerpt,
        relevance_score=float(score) if score is not None else None,
    )


def _score_for_sorting(passage: RetrievedPassage) -> float:
    return passage.relevance_score if passage.relevance_score is not None else float("-inf")


def _embedding_values(response: Any) -> list[float]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if not embeddings:
        raise ValueError("Vertex returned no embedding")
    first = embeddings[0]
    values = getattr(first, "values", None)
    if values is None and isinstance(first, dict):
        values = first.get("values")
    if values is None:
        raise ValueError("Vertex returned an embedding without values")
    return [float(value) for value in values]
