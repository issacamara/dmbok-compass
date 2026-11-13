"""A narrow, content-free boundary for operational telemetry.

Production interactions are deliberately not accepted by this module. Callers
provide only typed response outcomes or exception classes, and the logging
adapter serializes the already allowlisted :class:`TelemetryEvent` fields.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Literal, Protocol

from app.contracts import AnswerResponse, TelemetryEvent, validate_telemetry_payload

logger = logging.getLogger("dmbok_compass.telemetry")


class TelemetryAdapter(Protocol):
    """Emit one already-sanitized operational event."""

    def emit(self, event: TelemetryEvent) -> None: ...


class LoggingTelemetryAdapter:
    """Write only the F2 allowlisted event, never exception text or payloads."""

    def emit(self, event: TelemetryEvent) -> None:
        payload = event.model_dump(exclude_none=True, mode="json")
        validate_telemetry_payload(payload)
        logger.info(json.dumps(payload, sort_keys=True))


class InMemoryTelemetryAdapter:
    """Test/local adapter that retains only typed, allowlisted events."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        validate_telemetry_payload(event.model_dump(exclude_none=True))
        self.events.append(event)


def response_event(response: AnswerResponse, started: float) -> TelemetryEvent:
    """Reduce an ephemeral response to content-free operational metadata."""

    return TelemetryEvent(
        request_count=1,
        outcome="error" if response.outcome == "operational_error" else response.outcome,
        duration_ms=_elapsed_ms(started),
        provider_model=response.trace.selected_model,
        model_attempts=response.trace.model_attempts or None,
    )


def error_event(error: BaseException, started: float | None = None) -> TelemetryEvent:
    """Represent an error by class only; exception messages are never logged."""

    return TelemetryEvent(
        request_count=1,
        outcome="error",
        duration_ms=_elapsed_ms(started) if started is not None else None,
        error_class=type(error).__name__,
    )


def stage_event(
    stage: Literal["parsing", "generation", "judging"],
    status: Literal["started", "succeeded", "failed", "awaiting_budget", "resumed"],
    *,
    completed_count: int | None = None,
    total_count: int | None = None,
    usage_kind: Literal["pages", "units", "credits", "tokens"] | None = None,
    usage_units: int | None = None,
    error: BaseException | None = None,
    http_status_code: int | None = None,
) -> TelemetryEvent:
    """Build a typed stage event without accepting content or provider errors."""

    return TelemetryEvent(
        event="stage_operation",
        stage=stage,
        stage_status=status,
        completed_count=completed_count,
        total_count=total_count,
        usage_kind=usage_kind,
        usage_units=usage_units,
        error_class=type(error).__name__ if error is not None else None,
        http_status_code=http_status_code,
    )


def _elapsed_ms(started: float) -> float:
    return round(max(0.0, (perf_counter() - started) * 1000), 3)
