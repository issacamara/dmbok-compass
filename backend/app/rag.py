"""Validated construction for every native LangChain RAG integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.corpus.embedding import create_vertex_embeddings
from app.corpus.indexing import (
    VECTOR_COLLECTION,
    CorpusEmbeddingWriter,
    create_vector_store,
)
from app.retrieval.search import FirestorePassageRetriever
from model.fallback import FallbackAdapter
from model.gemini import create_gemini_adapter
from model.openrouter import OpenRouterAdapter, parse_json_schema_supported
from model.protocol import ModelAdapter


@dataclass(frozen=True, slots=True)
class LangChainRagSettings:
    project: str
    location: str = "europe-west1"
    vector_collection: str = VECTOR_COLLECTION
    answer_provider: str = "openrouter"

    @classmethod
    def from_environment(cls) -> "LangChainRagSettings":
        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
            or ""
        ).strip()
        if not project:
            raise ValueError("The Google Cloud project is not configured.")
        settings = cls(
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1").strip(),
            vector_collection=os.environ.get(
                "RAG_VECTOR_COLLECTION", VECTOR_COLLECTION
            ).strip(),
            answer_provider=os.environ.get("ANSWER_PROVIDER", "openrouter").strip().lower(),
        )
        if not settings.location or not settings.vector_collection:
            raise ValueError("LangChain location and vector collection must be configured.")
        if settings.answer_provider not in {"openrouter", "vertex"}:
            raise ValueError("The answer provider is not configured.")
        return settings


class LangChainRagFactory:
    """Build consistent ingestion, retrieval, and generation integrations."""

    def __init__(self, client: Any, settings: LangChainRagSettings) -> None:
        self.client = client
        self.settings = settings
        self._embeddings = create_vertex_embeddings(
            project=settings.project, location=settings.location
        )
        self._vector_store = create_vector_store(
            client,
            self._embeddings,
            collection_id=settings.vector_collection,
        )

    @classmethod
    def from_environment(cls, client: Any) -> "LangChainRagFactory":
        return cls(client, LangChainRagSettings.from_environment())

    def ingestion_writer(self, *, source_metadata: dict[str, Any]) -> CorpusEmbeddingWriter:
        return CorpusEmbeddingWriter(
            self._vector_store,
            source_metadata=source_metadata,
        )

    def retriever(self) -> FirestorePassageRetriever:
        return FirestorePassageRetriever(self._vector_store)

    def generation_adapter(self) -> ModelAdapter:
        if self.settings.answer_provider == "openrouter":
            return OpenRouterAdapter.from_environment()
        vertex = create_gemini_adapter(
            project=self.settings.project,
            location=self.settings.location,
            vertexai=True,
            primary_model=os.environ.get(
                "VERTEX_PRIMARY_MODEL", "gemini-2.5-flash-lite"
            ),
            fallback_model=None,
        )
        secondary = OpenRouterAdapter(
            os.environ.get("OPENROUTER_API_KEY", ""),
            primary_model=os.environ.get(
                "OPENROUTER_PRIMARY_MODEL", "mistralai/mistral-nemo"
            ),
            fallback_model=None,
            json_schema_supported=parse_json_schema_supported(
                os.environ.get("OPENROUTER_JSON_SCHEMA_SUPPORTED", "true")
            ),
        )
        return FallbackAdapter(vertex, secondary)

    def selected_model(self) -> str:
        if self.settings.answer_provider == "openrouter":
            return os.environ.get("OPENROUTER_PRIMARY_MODEL", "answer-model")
        return os.environ.get("VERTEX_PRIMARY_MODEL", "gemini-2.5-flash-lite")
