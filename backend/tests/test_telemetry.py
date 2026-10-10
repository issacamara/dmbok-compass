import logging
from time import perf_counter

from app.contracts.api import AnswerResponse, ModelAttempt, QuotaStatus, RetrievalTrace
from app.telemetry import InMemoryTelemetryAdapter, LoggingTelemetryAdapter, error_event, response_event


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at="2030-01-01T00:00:00Z",
    )


def answer() -> AnswerResponse:
    return AnswerResponse(
        outcome="answer",
        answer_text="CANARY ANSWER",
        trace=RetrievalTrace(
            retrieved_passages=[],
            selected_model="test-model",
            timings_ms={"total_ms": 2},
        ),
        quota=quota(),
    )


def test_response_and_error_events_are_content_free() -> None:
    response = response_event(answer(), perf_counter())
    error = error_event(ValueError("CANARY QUESTION / PASSAGE"))

    assert response.model_dump(exclude_none=True) == {
        "request_count": 1,
        "outcome": "answer",
        "duration_ms": response.duration_ms,
        "provider_model": "test-model",
    }
    assert error.model_dump(exclude_none=True)["error_class"] == "ValueError"
    assert "CANARY" not in str(response.model_dump())
    assert "CANARY" not in str(error.model_dump())


def test_logging_adapter_never_logs_interaction_payload(caplog) -> None:
    caplog.set_level(logging.INFO, logger="dmbok_compass.telemetry")
    LoggingTelemetryAdapter().emit(response_event(answer(), perf_counter()))

    rendered = " ".join(record.getMessage() for record in caplog.records)
    assert "CANARY" not in rendered
    assert "answer" in rendered
    assert "test-model" in rendered


def test_in_memory_adapter_keeps_only_typed_events() -> None:
    adapter = InMemoryTelemetryAdapter()
    adapter.emit(response_event(answer(), perf_counter()))

    assert len(adapter.events) == 1
    assert adapter.events[0].outcome == "answer"
    assert set(adapter.events[0].model_dump(exclude_none=True)) <= {
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
    }


def test_provider_failure_telemetry_records_only_route_metadata() -> None:
    failed = answer().model_copy(update={
        "trace": RetrievalTrace(
            retrieved_passages=[], selected_model="fallback",
            model_attempts=[
                ModelAttempt(model="primary", outcome="unavailable", status_code=404),
                ModelAttempt(model="fallback", outcome="unavailable", status_code=503),
            ],
        )
    })
    event = response_event(failed, perf_counter()).model_dump(exclude_none=True)
    assert [item["status_code"] for item in event["model_attempts"]] == [404, 503]
    assert "CANARY" not in str(event)
