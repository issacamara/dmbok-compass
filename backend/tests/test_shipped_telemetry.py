"""Preview verification for the shipped log -> metric -> notification path.

The local procedure is: load ``app/logging_config.json`` with ``dictConfig``,
emit answer/refusal/error fixtures, parse stdout as JSON, and assert each
Terraform filter's typed marker is present. The real preview gate repeats this
with a deployed image and a configured notification channel; no local test
claims delivered-alert evidence.
"""

import json
import logging.config
from pathlib import Path

from app.contracts.api import AnswerResponse, QuotaStatus, RetrievalTrace
from app.telemetry import LoggingTelemetryAdapter, error_event, response_event

ROOT = Path(__file__).parents[2]
BACKEND = Path(__file__).parents[1]
CONFIG_PATH = BACKEND / "app" / "logging_config.json"
TERRAFORM_PATH = ROOT / "infra" / "terraform" / "monitoring.tf"


def response(outcome: str) -> AnswerResponse:
    return AnswerResponse(
        outcome=outcome,
        answer_text="CANARY ANSWER" if outcome != "refusal" else "CANARY REFUSAL",
        trace=RetrievalTrace(retrieved_passages=[], selected_model="fallback-model"),
        quota=QuotaStatus(
            user_used=1,
            user_limit=100,
            global_used=1,
            global_limit=100,
            resets_at="2030-01-01T00:00:00Z",
        ),
    )


def test_shipped_logger_emits_json_for_success_refusal_and_failure(capsys) -> None:
    root = logging.getLogger()
    telemetry = logging.getLogger("dmbok_compass.telemetry")
    root_handlers = root.handlers[:]
    telemetry_handlers = telemetry.handlers[:]
    telemetry_level = telemetry.level
    telemetry_propagate = telemetry.propagate
    try:
        logging.config.dictConfig(json.loads(CONFIG_PATH.read_text()))
        adapter = LoggingTelemetryAdapter()
        adapter.emit(response_event(response("answer"), 0))
        adapter.emit(response_event(response("refusal"), 0))
        adapter.emit(error_event(ValueError("CANARY QUESTION / PASSAGE"), 0))

        lines = [line for line in capsys.readouterr().out.splitlines() if line]
        events = [json.loads(line) for line in lines]
        assert {event["outcome"] for event in events} == {"answer", "refusal", "error"}
        assert all("CANARY" not in line for line in lines)
    finally:
        root.handlers = root_handlers
        telemetry.handlers = telemetry_handlers
        telemetry.setLevel(telemetry_level)
        telemetry.propagate = telemetry_propagate


def test_terraform_filters_match_emitted_typed_fields_and_unknown_usage() -> None:
    terraform = TERRAFORM_PATH.read_text()
    error_event_payload = error_event(ValueError(), 0).model_dump(exclude_none=True)
    fallback_event_payload = response_event(response("answer"), 0).model_dump(exclude_none=True)

    filters = [line for line in terraform.splitlines() if "filter" in line]
    assert any('jsonPayload.outcome' in line and 'error' in line for line in filters)
    assert any('jsonPayload.provider_model' in line and 'fallback_model' in line for line in filters)
    assert error_event_payload["outcome"] == "error"
    assert fallback_event_payload["provider_model"] == "fallback-model"
    assert "token_estimate" not in fallback_event_payload


def test_shipped_config_enables_telemetry_without_content_handlers() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    telemetry = config["loggers"]["dmbok_compass.telemetry"]
    assert telemetry == {"handlers": ["telemetry"], "level": "INFO", "propagate": False}
    assert config["handlers"]["telemetry"]["stream"] == "ext://sys.stdout"
