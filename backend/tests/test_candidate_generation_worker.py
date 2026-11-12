from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import pytest

import app.corpus.staging as staging_module
import app.evaluation.automatic as automatic_module
import app.evaluation.candidate_generation as generation_module
import app.evaluation.dispatch as dispatch_module
import app.evaluation.worker as worker_module
import app.provider_allowance_firestore as allowance_module
import app.provider_allowance as allowance_contract_module
import app.settings as settings_module
import app.telemetry as telemetry_module
import model.candidate_generation as candidate_model_module


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "backend" / "app" / "logging_config.json"
ATTEMPT_ID = "attempt-364"
CORPUS_ID = "corpus-364"


class StagingStore:
    def __init__(self, _client, *, cleanup_failure=None, cleanup_calls=None) -> None:
        self.attempt = {
            "attempt_id": ATTEMPT_ID,
            "state": "staged",
            "staged_corpus_version_id": CORPUS_ID,
        }
        self.updates: list[dict[str, object]] = []
        self.cleanup_failure = cleanup_failure
        self.cleanup_calls = cleanup_calls if cleanup_calls is not None else []

    def get(self, *, attempt_id: str):
        assert attempt_id == ATTEMPT_ID
        return dict(self.attempt)

    def update_candidate_generation_status(self, **values) -> None:
        self.updates.append(values)
        if values["state"] == "failed":
            self.cleanup_calls.append("status")
            if self.cleanup_failure == "status":
                raise RuntimeError("CANARY STATUS CLEANUP")
        self.attempt["candidate_generation_state"] = values["state"]
        for key, value in (
            ("candidate_generation_count", values.get("completed_count")),
            ("candidate_generation_target", values.get("target_count")),
            ("candidate_generation_rejected_count", values.get("rejected_count")),
            ("candidate_generation_usage_tokens", values.get("usage_tokens")),
        ):
            if value is not None:
                self.attempt[key] = value


def _prepare_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    committed_head=None,
    committed_rows=(),
    previous_progress: dict[str, object] | None = None,
    generation_error: Exception | None = None,
    cleanup_failure: str | None = None,
):
    stage = SimpleNamespace(
        route="vertex", model="candidate-model", configuration_version="cfg-v1",
        allowance_policy="vertex-project", allowance_max_units=50,
    )
    runtime = SimpleNamespace(
        generation_timeout_ms=1_000, output_token_cap=128,
        vector_collection="candidate-vectors", runtime=SimpleNamespace(location="us-central1"),
    )
    cleanup_calls: list[str] = []
    staging = StagingStore(None, cleanup_failure=cleanup_failure, cleanup_calls=cleanup_calls)
    staging.attempt.update(previous_progress or {})
    def release_claim(*_args, **_kwargs):
        cleanup_calls.append("claim")
        if cleanup_failure == "claim":
            raise RuntimeError("CANARY CLAIM CLEANUP")

    claims = SimpleNamespace(release=release_claim)
    allowances = SimpleNamespace(stage_reservation=lambda *_args, **_kwargs: SimpleNamespace(status="staged"))

    class CandidateDispatcher:
        @staticmethod
        def operation_key(*_args):
            return "candidate-operation"

        @staticmethod
        def claim_id(_operation_key):
            return "candidate-claim"

    class CandidatePublisher:
        def __init__(self, *_args, **_kwargs):
            pass

        def launch(self, **_kwargs):
            return None

    class GenerationStore:
        def __init__(self, _client):
            pass

        def operation_head(self, *_args, **_kwargs):
            return committed_head

        def get_revision(self, head):
            assert head is committed_head
            return tuple(committed_rows)

        def claim_operation(self, *_args, **_kwargs):
            return True

        def release_operation(self, *_args, **_kwargs):
            cleanup_calls.append("operation")
            if cleanup_failure == "operation":
                raise RuntimeError("CANARY OPERATION CLEANUP")
            return None

        def get_parse_derived_records(self, **_kwargs):
            return (), ()

    class AllowanceCapabilities:
        def __init__(self, **_kwargs):
            pass

    class StageConfiguration:
        @staticmethod
        def from_environment(_stage):
            return stage

    class CandidateAdapter:
        def __init__(self, **_kwargs):
            pass

    class CorpusStore:
        def __init__(self, *_args, **_kwargs):
            pass

    async def generate(**_kwargs):
        if generation_error is not None:
            raise generation_error
        raise AssertionError("generation is not expected in a committed-head replay")

    original_telemetry_adapter = telemetry_module.LoggingTelemetryAdapter

    class TestTelemetryAdapter(original_telemetry_adapter):
        def emit(self, event) -> None:
            if event.stage_status == "failed":
                cleanup_calls.append("telemetry")
                if cleanup_failure == "telemetry":
                    raise RuntimeError("CANARY TELEMETRY CLEANUP")
            super().emit(event)

    monkeypatch.setattr(staging_module, "FirestoreCorpusStagingAttemptStore", lambda _client: staging)
    monkeypatch.setattr(automatic_module, "CandidateExplorationPublisher", CandidatePublisher)
    monkeypatch.setattr(automatic_module, "candidate_generator_fingerprint", lambda *_args: "fingerprint-v1")
    monkeypatch.setattr(dispatch_module, "CloudRunCandidateGenerationDispatcher", CandidateDispatcher)
    monkeypatch.setattr(dispatch_module, "EvaluationDispatchClaimStore", lambda _client: claims)
    monkeypatch.setattr(worker_module, "FirestoreEvaluationGenerationStore", GenerationStore)
    monkeypatch.setattr(worker_module, "FirestoreCorpusVersionStore", CorpusStore)
    monkeypatch.setattr(worker_module, "_configured_evaluation_service", lambda: object())
    monkeypatch.setattr(allowance_module, "FirestoreProviderAllowanceStore", lambda _client: allowances)
    monkeypatch.setattr(allowance_contract_module, "ProviderAccountCapabilities", AllowanceCapabilities)
    monkeypatch.setattr(settings_module, "EvaluationStageConfiguration", StageConfiguration)
    monkeypatch.setattr(telemetry_module, "LoggingTelemetryAdapter", TestTelemetryAdapter)
    monkeypatch.setattr(candidate_model_module, "EvaluationCandidateAdapter", CandidateAdapter)
    monkeypatch.setattr(generation_module, "generate_candidate_revision", generate)
    monkeypatch.setenv("CORPUS_STAGING_ATTEMPT_ID", ATTEMPT_ID)
    monkeypatch.setenv("EVALUATION_CORPUS_VERSION_ID", CORPUS_ID)
    monkeypatch.setenv("EVALUATION_DISPATCH_OWNER_ID", "dispatch-owner")
    monkeypatch.setenv("EVALUATION_GENERATOR_FINGERPRINT", "fingerprint-v1")
    return staging, runtime


def _with_json_telemetry(capsys, action) -> tuple[str, str]:
    root = logging.getLogger()
    telemetry = logging.getLogger("dmbok_compass.telemetry")
    root_handlers = root.handlers[:]
    telemetry_handlers = telemetry.handlers[:]
    telemetry_level = telemetry.level
    telemetry_propagate = telemetry.propagate
    try:
        logging.config.dictConfig(json.loads(CONFIG_PATH.read_text()))
        action()
        captured = capsys.readouterr()
        return captured.out, captured.err
    finally:
        root.handlers = root_handlers
        telemetry.handlers = telemetry_handlers
        telemetry.setLevel(telemetry_level)
        telemetry.propagate = telemetry_propagate


def test_committed_candidate_head_replay_restores_terminal_status_and_success_event(monkeypatch, capsys) -> None:
    head = object()
    previous = {
        "candidate_generation_state": "running",
        "candidate_generation_count": 42,
        "candidate_generation_target": 200,
        "candidate_generation_rejected_count": 5,
        "candidate_generation_usage_tokens": 1_640,
    }
    staging, runtime = _prepare_fakes(
        monkeypatch, committed_head=head, committed_rows=range(42), previous_progress=previous
    )

    output, _stderr = _with_json_telemetry(
        capsys,
        lambda: worker_module._prepare_candidate_evaluation(object(), "project", runtime),
    )

    status = staging.updates[-1]
    assert status == {
        "attempt_id": ATTEMPT_ID,
        "state": "published",
        "completed_count": 42,
        "target_count": 200,
        "rejected_count": 5,
        "usage_tokens": 1_640,
    }
    event = json.loads(output.splitlines()[-1])
    assert event["stage"] == "generation"
    assert event["stage_status"] == "succeeded"
    assert event["completed_count"] == 42
    assert event["total_count"] == 200
    assert event["usage_kind"] == "tokens"
    assert event["usage_units"] == 1_640


def test_committed_candidate_replay_keeps_unrecorded_usage_unknown(monkeypatch, capsys) -> None:
    head = object()
    staging, runtime = _prepare_fakes(
        monkeypatch, committed_head=head, committed_rows=range(42),
        previous_progress={"candidate_generation_state": "running"},
    )

    output, _stderr = _with_json_telemetry(
        capsys,
        lambda: worker_module._prepare_candidate_evaluation(object(), "project", runtime),
    )

    status = staging.updates[-1]
    assert status["state"] == "published"
    assert status["completed_count"] == 42
    assert status["target_count"] == 200
    assert status["rejected_count"] is None
    assert status["usage_tokens"] is None
    assert "candidate_generation_rejected_count" not in staging.attempt
    assert "candidate_generation_usage_tokens" not in staging.attempt
    event = json.loads(output.splitlines()[-1])
    assert event["stage_status"] == "succeeded"
    assert event["completed_count"] == 42
    assert event["total_count"] == 200
    assert "usage_kind" not in event
    assert "usage_units" not in event


def test_committed_candidate_replay_target_never_falls_below_rows(monkeypatch, capsys) -> None:
    head = object()
    staging, runtime = _prepare_fakes(
        monkeypatch,
        committed_head=head,
        committed_rows=range(225),
        previous_progress={"candidate_generation_state": "running"},
    )

    output, _stderr = _with_json_telemetry(
        capsys,
        lambda: worker_module._prepare_candidate_evaluation(object(), "project", runtime),
    )

    status = staging.updates[-1]
    assert status["completed_count"] == 225
    assert status["target_count"] == 225
    event = json.loads(output.splitlines()[-1])
    assert event["completed_count"] == 225
    assert event["total_count"] == 225


@pytest.mark.parametrize("cleanup_failure", [None, "operation", "claim", "status", "telemetry"])
def test_candidate_generation_failure_cleanup_is_best_effort_and_sanitized(
    monkeypatch, capsys, cleanup_failure
) -> None:
    staging, runtime = _prepare_fakes(
        monkeypatch,
        generation_error=RuntimeError("CANARY PROVIDER BODY and credential detail"),
        cleanup_failure=cleanup_failure,
    )

    def action() -> None:
        with pytest.raises(RuntimeError, match="candidate dataset generation failed") as raised:
            worker_module._prepare_candidate_evaluation(object(), "project", runtime)
        traceback.print_exception(raised.value, file=sys.stderr)
        assert raised.value.__suppress_context__ is True

    output, stderr = _with_json_telemetry(capsys, action)
    assert "CANARY" not in output
    assert "CANARY" not in stderr
    assert "candidate dataset generation failed" in stderr
    assert [call for call in staging.cleanup_calls] == ["operation", "claim", "status", "telemetry"]
    if cleanup_failure == "status":
        assert staging.attempt["candidate_generation_state"] == "running"
    else:
        assert staging.attempt["candidate_generation_state"] == "failed"
