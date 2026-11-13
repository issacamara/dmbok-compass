"""Allowlisted, content-free operational telemetry."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    "event",
    "stage",
    "stage_status",
    "completed_count",
    "total_count",
    "usage_kind",
    "usage_units",
    "http_status_code",
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
    event: Literal["stage_operation"] | None = None
    stage: Literal["parsing", "generation", "judging"] | None = None
    stage_status: Literal["started", "succeeded", "failed", "awaiting_budget", "resumed"] | None = None
    completed_count: int | None = Field(default=None, ge=0)
    total_count: int | None = Field(default=None, ge=0)
    usage_kind: Literal["pages", "units", "credits", "tokens"] | None = None
    usage_units: int | None = Field(default=None, ge=0)
    http_status_code: int | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="after")
    def validate_stage_event(self) -> "TelemetryEvent":
        if self.event is None:
            if any(value is not None for value in (self.stage, self.stage_status, self.completed_count, self.total_count, self.usage_kind, self.usage_units)):
                raise ValueError("stage telemetry fields require a stage event")
            return self
        if self.stage is None or self.stage_status is None:
            raise ValueError("stage events require a stage and status")
        if (self.completed_count is None) != (self.total_count is None):
            raise ValueError("stage progress requires both completed and total counts")
        if self.usage_units is not None and self.usage_kind is None:
            raise ValueError("reported usage requires a unit kind")
        if self.usage_kind is not None and self.usage_units is None:
            raise ValueError("unknown usage must omit its unit kind")
        if self.completed_count is not None and self.total_count is not None and self.completed_count > self.total_count:
            raise ValueError("completed stage count cannot exceed total count")
        return self


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
