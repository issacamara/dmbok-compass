"""Native LangChain vector-store writes for immutable corpus chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore

from app.contracts import ChildDocumentChunk, ParentChunk
from app.corpus.chunking import TextChunk

VECTOR_COLLECTION = "document_chunks_v2"
VECTOR_SCHEMA_VERSION = "2"
PARENT_COLLECTION = "corpus_parent_chunks"


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

    max_batch_size = 500

    def write(self, chunks: Sequence[TextChunk]) -> tuple[TextChunk, ...]:
        """Persist deterministic chunk IDs in Firestore-safe batches."""

        stored: list[TextChunk] = []
        for start in range(0, len(chunks), self.max_batch_size):
            stored.extend(self.write_batch(chunks[start : start + self.max_batch_size]))
        return tuple(stored)

    def write_batch(self, chunks: Sequence[TextChunk]) -> tuple[TextChunk, ...]:
        """Persist one explicit-ID batch without activating its corpus version."""

        if len(chunks) > self.max_batch_size:
            raise ValueError(f"corpus write batches may contain at most {self.max_batch_size} chunks")
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


class HierarchicalCorpusWriter(CorpusEmbeddingWriter):
    """Persist unembedded parents and explicitly identified embedded children."""

    def __init__(
        self,
        vector_store: FirestoreVectorStore,
        parent_store: Any,
        *,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(vector_store, source_metadata=source_metadata)
        self.parent_store = parent_store

    def write_parent_batch(self, parents: Sequence[ParentChunk]) -> tuple[ParentChunk, ...]:
        if len(parents) > self.max_batch_size:
            raise ValueError(f"corpus write batches may contain at most {self.max_batch_size} parents")
        self.parent_store.write_batch(parents, metadata=self.source_metadata)
        return tuple(parents)

    def write_child_batch(
        self, children: Sequence[ChildDocumentChunk]
    ) -> tuple[ChildDocumentChunk, ...]:
        if len(children) > self.max_batch_size:
            raise ValueError(f"corpus write batches may contain at most {self.max_batch_size} children")
        documents = [
            Document(
                page_content=child.text,
                metadata={
                    **self.source_metadata,
                    "schema_version": child.schema_version,
                    "corpus_version_id": child.corpus_version_id,
                    "chunk_id": child.chunk_id,
                    "parent_id": child.parent_id,
                    "section_path": list(child.section_path),
                    "page_start": child.page_start,
                    "page_end": child.page_end,
                    "source_spans": [span.model_dump(mode="json", exclude_none=True) for span in child.source_spans],
                    **({"source_text": child.source_text} if child.source_text is not None else {}),
                    "child_ordinal": child.child_ordinal,
                    "content_hash": child.content_hash,
                    "token_count": child.token_count,
                },
            )
            for child in children
        ]
        ids = [child.chunk_id for child in children]
        stored_ids = self.vector_store.add_documents(documents, ids=ids)
        if list(stored_ids) != ids:
            raise RuntimeError("LangChain vector store did not preserve deterministic child IDs")
        return tuple(children)
