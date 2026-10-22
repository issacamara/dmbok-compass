"""Fail-closed, immutable runtime configuration loaded from the environment."""

from __future__ import annotations

import os
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.contracts import RuntimeSettings


class RuntimeConfiguration(BaseModel):
    """Validated non-secret settings shared by every runtime entry point."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    runtime: RuntimeSettings
    vector_collection: str = Field(min_length=1, max_length=256)
    vector_schema_version: str = Field(min_length=1, max_length=32)
    embedding_model: str = Field(min_length=1, max_length=128)
    embedding_dimensions: int = Field(ge=1, le=4096)
    retrieval_strategy: str = Field(pattern=r"^(dense|hybrid|reranked)$")
    request_timeout_ms: int = Field(ge=1, le=60_000)
    generation_timeout_ms: int = Field(ge=1, le=60_000)
    output_token_cap: int = Field(ge=1, le=32_000)
    max_model_attempts: int = Field(ge=1, le=2)
    json_schema_supported: bool

    @model_validator(mode="after")
    def validate_combination(self) -> "RuntimeConfiguration":
        if self.generation_timeout_ms > self.request_timeout_ms:
            raise ValueError("generation timeout must not exceed request timeout")
        if self.runtime.provider == "vertex" and self.runtime.fallback_model is not None:
            raise ValueError("vertex route does not support a fallback model")
        if self.runtime.provider == "openrouter" and self.runtime.fallback_model is None:
            raise ValueError("openrouter route requires a fallback model")
        if self.retrieval_strategy != "dense":
            raise ValueError("only the dense retrieval strategy is enabled")
        return self

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfiguration":
        env = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise ValueError(f"Missing required runtime setting: {name}")
            return value

        provider = required("ANSWER_PROVIDER").lower()
        if provider not in {"openrouter", "vertex"}:
            raise ValueError("ANSWER_PROVIDER must be openrouter or vertex")
        if provider == "openrouter" and not env.get("OPENROUTER_API_KEY", "").strip():
            raise ValueError("Missing required runtime credential: OPENROUTER_API_KEY")

        try:
            runtime = RuntimeSettings(
                project=required("GOOGLE_CLOUD_PROJECT"),
                location=required("GOOGLE_CLOUD_LOCATION"),
                provider=provider,
                primary_model=required(
                    "OPENROUTER_PRIMARY_MODEL" if provider == "openrouter" else "VERTEX_PRIMARY_MODEL"
                ),
                fallback_model=(env.get("OPENROUTER_FALLBACK_MODEL", "").strip() or None)
                if provider == "openrouter"
                else None,
                secret_version=required("RUNTIME_SECRET_VERSION"),
                lexical_artifact_id=env.get("LEXICAL_ARTIFACT_ID", "").strip() or None,
            )
            return cls(
                runtime=runtime,
                vector_collection=required("RAG_VECTOR_COLLECTION"),
                vector_schema_version=required("RAG_VECTOR_SCHEMA_VERSION"),
                embedding_model=required("RAG_EMBEDDING_MODEL"),
                embedding_dimensions=int(required("RAG_EMBEDDING_DIMENSIONS")),
                retrieval_strategy=required("RAG_RETRIEVAL_STRATEGY").lower(),
                request_timeout_ms=int(required("ANSWER_REQUEST_TIMEOUT_MS")),
                generation_timeout_ms=int(required("ANSWER_GENERATION_TIMEOUT_MS")),
                output_token_cap=int(required("ANSWER_OUTPUT_TOKEN_CAP")),
                max_model_attempts=int(required("ANSWER_MAX_MODEL_ATTEMPTS")),
                json_schema_supported=_required_bool(env, "OPENROUTER_JSON_SCHEMA_SUPPORTED"),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                raise ValueError("Invalid runtime settings: " + "; ".join(error["msg"] for error in exc.errors())) from exc
            raise ValueError(f"Invalid runtime settings: {exc}") from exc

    def public_settings(self) -> RuntimeSettings:
        """Return the reference-only frozen contract; never expose credentials."""

        return self.runtime


def _required_bool(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name, "").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def startup_configuration(environ: Mapping[str, str] | None = None) -> RuntimeConfiguration:
    """Load configuration once at process startup with safe diagnostics."""

    try:
        return RuntimeConfiguration.from_environment(environ)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None
