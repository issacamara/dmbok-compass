"""Validated construction for every native LangChain RAG integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.contracts import CandidateIdentity, CorpusServingBinding, EvaluationCandidateSnapshot
from app.settings import RuntimeConfiguration
from app.corpus.embedding import create_vertex_embeddings
from app.corpus.indexing import (
    VECTOR_COLLECTION,
    CorpusEmbeddingWriter,
    create_vector_store,
)
from app.retrieval.search import FirestorePassageRetriever
from app.retrieval.hybrid import HybridRetriever, HybridStrategy
from app.retrieval.lexical import LexicalIndex
from model.fallback import FallbackAdapter
from model.gemini import create_gemini_adapter
from model.langchain import enforce_private_langchain_runtime
from model.openrouter import OpenRouterAdapter, parse_json_schema_supported
from model.protocol import ModelAdapter


@dataclass(frozen=True, slots=True)
class LangChainRagSettings:
    project: str
    location: str = "europe-west1"
    vector_collection: str = VECTOR_COLLECTION
    answer_provider: str = "openrouter"
    primary_model: str = "meta-llama/llama-3.1-8b-instruct"
    fallback_model: str | None = "google/gemma-4-26b-a4b-it:free"
    json_schema_supported: bool = True

    @classmethod
    def from_environment(cls) -> "LangChainRagSettings":
        configuration = RuntimeConfiguration.from_environment()
        runtime = configuration.runtime
        return cls(
            project=runtime.project,
            location=runtime.location,
            vector_collection=configuration.vector_collection,
            answer_provider=runtime.provider,
            primary_model=runtime.primary_model,
            fallback_model=runtime.fallback_model,
            json_schema_supported=configuration.json_schema_supported,
        )


class LangChainRagFactory:
    """Build consistent ingestion, retrieval, and generation integrations."""

    def __init__(self, client: Any, settings: LangChainRagSettings) -> None:
        enforce_private_langchain_runtime()
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

    def experimental_hybrid_retriever(
        self,
        lexical_index: LexicalIndex,
        candidate: CandidateIdentity,
        *,
        strategy: HybridStrategy = HybridStrategy(),
    ) -> HybridRetriever:
        """Build the opt-in experiment without changing dense serving."""

        return HybridRetriever(
            self.retriever().retrieve,
            lexical_index,
            candidate,
            strategy=strategy,
        )

    def retriever_for_binding(self, binding: CorpusServingBinding) -> FirestorePassageRetriever:
        """Create a native store against one immutable serving binding."""

        if binding.vector_schema_version != "2":
            raise ValueError("unsupported corpus vector schema")
        if binding.embedding_dimensions != 768:
            raise ValueError("unsupported corpus embedding dimensions")
        if binding.embedding_model != "gemini-embedding-001":
            raise ValueError("unsupported corpus embedding model")
        return FirestorePassageRetriever(
            create_vector_store(
                self.client,
                self._embeddings,
                collection_id=binding.vector_collection,
            )
        )

    def generation_adapter(self) -> ModelAdapter:
        if self.settings.answer_provider == "openrouter":
            return OpenRouterAdapter(
                os.environ["OPENROUTER_API_KEY"],
                primary_model=self.settings.primary_model,
                fallback_model=self.settings.fallback_model,
                json_schema_supported=self.settings.json_schema_supported,
            )
        vertex = create_gemini_adapter(
            project=self.settings.project,
            location=self.settings.location,
            vertexai=True,
            primary_model=self.settings.primary_model,
            fallback_model=None,
        )
        secondary = OpenRouterAdapter(
            os.environ["OPENROUTER_API_KEY"],
            primary_model=self.settings.fallback_model or self.settings.primary_model,
            fallback_model=None,
            json_schema_supported=parse_json_schema_supported(
                str(self.settings.json_schema_supported)
            ),
        )
        return FallbackAdapter(vertex, secondary)

    def generation_adapter_for_snapshot(
        self, snapshot: EvaluationCandidateSnapshot
    ) -> ModelAdapter:
        """Build generation with a run's recorded route, never current model env."""

        provider = snapshot.answer_provider
        primary_model = snapshot.primary_model
        fallback_model = snapshot.fallback_model
        if provider == "openrouter":
            return OpenRouterAdapter(
                os.environ["OPENROUTER_API_KEY"],
                primary_model=primary_model,
                fallback_model=fallback_model,
                json_schema_supported=snapshot.json_schema_supported,
            )
        vertex = create_gemini_adapter(
            project=self.settings.project,
            location=self.settings.location,
            vertexai=True,
            primary_model=primary_model,
            fallback_model=None,
        )
        secondary = OpenRouterAdapter(
            os.environ.get("OPENROUTER_API_KEY", ""),
            primary_model=fallback_model or "meta-llama/llama-3.1-8b-instruct",
            fallback_model=None,
                json_schema_supported=snapshot.json_schema_supported,
        )
        return FallbackAdapter(vertex, secondary)

    def candidate_snapshot(
        self,
        binding: CorpusServingBinding,
        *,
        request_timeout_ms: int,
        generation_timeout_ms: int,
    ) -> EvaluationCandidateSnapshot:
        """Capture all non-secret inputs needed to reconstruct an evaluation."""

        if binding.vector_collection != self.settings.vector_collection:
            raise ValueError("candidate corpus binding does not match the configured vector collection")
        if self.settings.answer_provider == "openrouter":
            primary = self.settings.primary_model
            fallback = self.settings.fallback_model
        else:
            primary = self.settings.primary_model
            fallback = self.settings.fallback_model
        return EvaluationCandidateSnapshot(
            corpus_binding=binding,
            project=self.settings.project,
            location=self.settings.location,
            answer_provider=self.settings.answer_provider,
            primary_model=primary,
            fallback_model=fallback or None,
            json_schema_supported=parse_json_schema_supported(
                str(self.settings.json_schema_supported)
            ),
            request_timeout_ms=request_timeout_ms,
            generation_timeout_ms=generation_timeout_ms,
            prompt_version="grounded-answer-policy-v1",
            retrieval_policy_version="top-five-cosine-v1",
            evidence_policy_version="grounded-evidence-v1",
            scorer_version="evaluation-gates-v1",
            dependency_lock_sha256=_dependency_lock_sha256(),
            integration_artifact="langchain-google-firestore==0.5.0.post2",
            runtime_version=os.environ.get("K_REVISION", "unversioned").strip() or "unversioned",
        )

    def selected_model(self) -> str:
        if self.settings.answer_provider == "openrouter":
            return self.settings.primary_model
        return self.settings.primary_model


def _dependency_lock_sha256() -> str:
    """Bind the canonical lock installed in the application image."""

    lock = Path(__file__).resolve().parents[1] / "requirements.lock"
    if not lock.is_file():
        raise ValueError("candidate dependency lock is unavailable")
    return sha256(lock.read_bytes()).hexdigest()
