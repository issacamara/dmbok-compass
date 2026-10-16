"""Validated construction for every native LangChain RAG integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.contracts import CorpusServingBinding, EvaluationCandidateSnapshot
from app.corpus.embedding import create_vertex_embeddings
from app.corpus.indexing import (
    VECTOR_COLLECTION,
    CorpusEmbeddingWriter,
    create_vector_store,
)
from app.retrieval.search import FirestorePassageRetriever
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

    def generation_adapter_for_snapshot(
        self, snapshot: EvaluationCandidateSnapshot
    ) -> ModelAdapter:
        """Build generation with a run's recorded route, never current model env."""

        provider = snapshot.answer_provider
        primary_model = snapshot.primary_model
        fallback_model = snapshot.fallback_model
        if provider == "openrouter":
            return OpenRouterAdapter(
                os.environ.get("OPENROUTER_API_KEY", ""),
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
            primary_model=fallback_model or "mistralai/mistral-nemo",
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
            primary = os.environ.get("OPENROUTER_PRIMARY_MODEL", "mistralai/mistral-nemo")
            fallback = os.environ.get(
                "OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.1-8b-instruct"
            )
        else:
            primary = os.environ.get("VERTEX_PRIMARY_MODEL", "gemini-2.5-flash-lite")
            fallback = os.environ.get("OPENROUTER_PRIMARY_MODEL", "mistralai/mistral-nemo")
        return EvaluationCandidateSnapshot(
            corpus_binding=binding,
            project=self.settings.project,
            location=self.settings.location,
            answer_provider=self.settings.answer_provider,
            primary_model=primary,
            fallback_model=fallback or None,
            json_schema_supported=parse_json_schema_supported(
                os.environ.get("OPENROUTER_JSON_SCHEMA_SUPPORTED", "true")
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
            return os.environ.get("OPENROUTER_PRIMARY_MODEL", "answer-model")
        return os.environ.get("VERTEX_PRIMARY_MODEL", "gemini-2.5-flash-lite")


def _dependency_lock_sha256() -> str:
    """Bind the packaged compatibility lock without persisting its contents."""

    lock = Path(__file__).resolve().parents[1] / "requirements-compat.lock"
    if not lock.is_file():
        raise ValueError("candidate dependency lock is unavailable")
    return sha256(lock.read_bytes()).hexdigest()
