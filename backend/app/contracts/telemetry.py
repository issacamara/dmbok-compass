"""Allowlisted, content-free operational telemetry."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from .api import ModelAttempt


TELEMETRY_FIELDS = (
    "request_count",
    "outcome",
    "duration_ms",
    "provider_model",
    "token_estimate",
    "quota_utilization",
    "corpus_version",
    "configuration_version",
    "error_class",
    "model_attempts",
)

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "question",
        "prompt",
        "passage",
        "passages",
        "answer",
        "answer_text",
        "citations",
        "trace",
        "retrieved_passages",
        "timings_ms",
        "quota",
        "excerpt",
        "password",
        "token",
        "secret",
        "credential",
    }
)


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_count: int | None = Field(default=None, ge=0)
    outcome: Literal["answer", "qualified", "refusal", "error"] | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    provider_model: str | None = Field(default=None, min_length=1, max_length=128)
    token_estimate: int | None = Field(default=None, ge=0)
    quota_utilization: float | None = Field(default=None, ge=0, le=1)
    corpus_version: str | None = Field(default=None, min_length=1, max_length=128)
    configuration_version: str | None = Field(default=None, min_length=1, max_length=128)
    error_class: str | None = Field(default=None, min_length=1, max_length=128)
    model_attempts: list[ModelAttempt] | None = Field(default=None, max_length=2)


def validate_telemetry_payload(payload: Mapping[str, Any]) -> None:
    """Reject forbidden content keys before telemetry serialization."""

    for key, value in payload.items():
        normalized = key.lower().replace("-", "_")
        if normalized in FORBIDDEN_FIELD_NAMES or normalized.rstrip("s") in FORBIDDEN_FIELD_NAMES:
            raise ValueError(f"forbidden telemetry field: {key}")
        if isinstance(value, Mapping):
            validate_telemetry_payload(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    validate_telemetry_payload(item)
