from __future__ import annotations

import json

import pytest

from app.settings import RuntimeConfiguration, startup_configuration


def environment() -> dict[str, str]:
    return {
        "GOOGLE_CLOUD_PROJECT": "dmbok-compass-dev",
        "GOOGLE_CLOUD_LOCATION": "europe-west1",
        "RAG_VECTOR_COLLECTION": "document_chunks_v2",
        "RAG_VECTOR_SCHEMA_VERSION": "2",
        "RAG_EMBEDDING_MODEL": "gemini-embedding-001",
        "RAG_EMBEDDING_DIMENSIONS": "768",
        "RAG_RETRIEVAL_STRATEGY": "dense",
        "ANSWER_PROVIDER": "openrouter",
        "ANSWER_REQUEST_TIMEOUT_MS": "14000",
        "ANSWER_GENERATION_TIMEOUT_MS": "8000",
        "ANSWER_OUTPUT_TOKEN_CAP": "1024",
        "ANSWER_MAX_MODEL_ATTEMPTS": "2",
        "RUNTIME_SECRET_VERSION": "openrouter-key-v3",
        "OPENROUTER_API_KEY": "raw-secret-must-not-escape",
        "OPENROUTER_PRIMARY_MODEL": "primary-model",
        "OPENROUTER_FALLBACK_MODEL": "fallback-model",
        "OPENROUTER_JSON_SCHEMA_SUPPORTED": "true",
    }


def test_runtime_configuration_is_immutable_and_secret_free() -> None:
    configuration = startup_configuration(environment())

    assert configuration.request_timeout_ms == 14_000
    assert configuration.generation_timeout_ms == 8_000
    assert configuration.public_settings().secret_version == "openrouter-key-v3"
    assert "raw-secret-must-not-escape" not in json.dumps(configuration.model_dump())
    with pytest.raises((TypeError, ValueError)):
        configuration.request_timeout_ms = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ANSWER_REQUEST_TIMEOUT_MS", "0"),
        ("ANSWER_GENERATION_TIMEOUT_MS", "16000"),
        ("RAG_EMBEDDING_DIMENSIONS", "0"),
        ("RAG_RETRIEVAL_STRATEGY", "hybrid"),
        ("ANSWER_MAX_MODEL_ATTEMPTS", "3"),
    ],
)
def test_invalid_runtime_combinations_fail_before_serving(name: str, value: str) -> None:
    env = environment()
    env[name] = value

    with pytest.raises(RuntimeError, match="Invalid runtime settings"):
        startup_configuration(env)


def test_missing_secret_reference_fails_with_safe_diagnostic() -> None:
    env = environment()
    del env["RUNTIME_SECRET_VERSION"]

    with pytest.raises(RuntimeError, match="RUNTIME_SECRET_VERSION") as caught:
        startup_configuration(env)
    assert "raw-secret" not in str(caught.value)


def test_vertex_configuration_requires_no_openrouter_fallback() -> None:
    env = environment()
    env.update(
        ANSWER_PROVIDER="vertex",
        VERTEX_PRIMARY_MODEL="gemini-2.5-flash-lite",
    )
    env.pop("OPENROUTER_API_KEY")
    env.pop("OPENROUTER_PRIMARY_MODEL")
    env.pop("OPENROUTER_FALLBACK_MODEL")

    configuration = RuntimeConfiguration.from_environment(env)
    assert configuration.runtime.provider == "vertex"
    assert configuration.runtime.fallback_model is None
