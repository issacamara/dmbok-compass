"""Native LangChain vector-store writes for immutable corpus chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore

from app.corpus.chunking import TextChunk

VECTOR_COLLECTION = "document_chunks_v2"
VECTOR_SCHEMA_VERSION = "2"


def create_vector_store(
    client: Any,
    embeddings: Any,
    *,
    collection_id: str = VECTOR_COLLECTION,
) -> FirestoreVectorStore:
    """Bind the native vector store to the v2 Firestore schema."""

    if not collection_id.strip():
        raise ValueError("collection_id must be non-empty")
    return FirestoreVectorStore(
        client.collection(collection_id),
        embeddings,
        client=client,
        content_field="text",
        metadata_field="metadata",
        embedding_field="embedding",
    )


class CorpusEmbeddingWriter:
    """Map deterministic chunks to Documents and let LangChain embed/write them."""

    def __init__(
        self,
        vector_store: FirestoreVectorStore,
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.source_metadata = dict(source_metadata or {})

    def write(self, chunks: Sequence[TextChunk]) -> tuple[TextChunk, ...]:
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    **self.source_metadata,
                    "schema_version": VECTOR_SCHEMA_VERSION,
                    "corpus_version_id": chunk.corpus_version_id,
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_ordinal": chunk.chunk_ordinal,
                    "content_hash": chunk.content_hash,
                },
            )
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        stored_ids = self.vector_store.add_documents(documents, ids=ids)
        if list(stored_ids) != ids:
            raise RuntimeError("LangChain vector store did not preserve deterministic chunk IDs")
        return tuple(chunks)
