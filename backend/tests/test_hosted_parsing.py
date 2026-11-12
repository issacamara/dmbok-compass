from __future__ import annotations

import hashlib
from threading import Event

import pytest

from app.ingestion.hosted_parsing import HostedParsingCoordinator, ParsingError, ParsingOutcomeUnknown
from app.ingestion.hosted_parsing import (
    CloudRunHostedParsingDispatcher,
    LlamaCloudParsingProvider,
    _parse_result,
)

ATTEMPT = "attempt-1"
PDF = b"%PDF-1.7\nexample"
SOURCE = {
    "bucket": "corpus-bucket",
    "object_name": f"staging/{ATTEMPT}/source.pdf",
    "generation": 12,
    "sha256": hashlib.sha256(PDF).hexdigest(),
}
RAW_RESULT = b'{"blocks":[{"block_id":"b1","physical_page":1,"kind":"text","raw_text":"Example"}]}'
RESULT = {
    "blocks": [{"block_id": "b1", "physical_page": 1, "kind": "text", "raw_text": "Example"}],
    "usage": {"pages": 1},
}


class Checkpoints:
    def __init__(self) -> None:
        self.records = {ATTEMPT: {"source": SOURCE}}

    def load(self, attempt_id):
        return self.records[attempt_id]

    def record_job(self, attempt_id, job_id, operation_key):
        self.records[attempt_id].update(remote_job_id=job_id, parsing_operation_key=operation_key)

    def mark_submission_started(self, attempt_id, operation_key):
        self.records[attempt_id].update(
            parsing_submission_started=True,
            parsing_operation_key=operation_key,
        )

    def record_artifact(self, attempt_id, artifact):
        self.records[attempt_id]["parsing_artifact"] = dict(artifact)

    def record_reservation(self, attempt_id, reservation):
        self.records[attempt_id]["allowance_reservation"] = dict(reservation)

    def record_status(
        self, attempt_id, state, *, error_code=None, page_count=None, block_count=None,
        usage_kind=None, usage_units=None,
    ):
        self.records[attempt_id].update(
            parsing_state=state, parsing_error_code=error_code,
            parsing_page_count=page_count, parsing_block_count=block_count,
            parsing_usage_kind=usage_kind, parsing_usage_units=usage_units,
        )


class Artifacts:
    def __init__(self):
        self.objects = {}
        self.writes = 0

    def get(self, path, generation=None):
        return self.objects[path]

    def put_if_absent(self, path, content, content_type):
        if path not in self.objects:
            self.objects[path] = content
            self.writes += 1
        return {"generation": 1}


class Provider:
    def __init__(self, *, statuses=None, unknown_submit=False, result=RESULT):
        self.jobs = {}
        self.statuses = list(statuses or ["succeeded"])
        self.unknown_submit = unknown_submit
        self.result_value = result
        self.submissions = 0
        self.result_calls = 0
        self.lookups = 0

    def submit(self, *, pdf, idempotency_key, route):
        self.submissions += 1
        if self.unknown_submit:
            raise TimeoutError("response lost")
        self.jobs[idempotency_key] = "job-1"
        return "job-1"

    def status(self, job_id):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def result(self, job_id):
        self.result_calls += 1
        return RAW_RESULT, self.result_value


def coordinator(provider, checkpoints, artifacts, *, route="llamacloud", reconcile=None, **kwargs):
    reservations, reconciled = [], []

    def reserve(key, amount):
        reservations.append((key, amount))
        return {"reservation_id": "reservation-1", "operation_key": key}

    parser = HostedParsingCoordinator(
        provider, checkpoints, artifacts,
        bucket_name="corpus-bucket", route=route,
        reserve_allowance=reserve,
        reconcile_allowance=reconcile or (lambda key, units: reconciled.append((key, units))),
        max_units=20, poll_interval_seconds=1, **kwargs,
    )
    return parser, reservations, reconciled


def test_invalid_pdf_and_malformed_schema_never_persist_artifact():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    parser, _, _ = coordinator(Provider(result={"blocks": [{"block_id": "bad"}]}), checkpoints, artifacts)
    with pytest.raises(ParsingError, match="not a PDF"):
        parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=b"not a PDF")
    assert parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF).state == "failed"
    assert artifacts.writes == 0


def test_restart_reconciles_saved_job_and_reuses_immutable_artifact():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider(statuses=["running", "running", "succeeded"])
    times = iter([0.0, 0.0, 2.0])
    parser, reservations, reconciled = coordinator(
        provider, checkpoints, artifacts, deadline_seconds=1,
        monotonic=lambda: next(times), sleep=lambda _: None,
    )
    assert parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF).state == "parsing"
    resumed, _, resumed_reconciled = coordinator(provider, checkpoints, artifacts)
    complete = resumed.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)
    repeated = resumed.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)

    assert complete.state == repeated.state == "awaiting_review"
    assert complete.artifact == repeated.artifact
    assert complete.artifact is not None
    assert complete.artifact.output_sha256 == hashlib.sha256(RAW_RESULT).hexdigest()
    assert complete.page_count == 1
    assert complete.block_count == 1
    assert complete.usage_kind == "pages"
    assert complete.usage_units == 1
    assert checkpoints.records[ATTEMPT]["parsing_page_count"] == 1
    assert checkpoints.records[ATTEMPT]["parsing_block_count"] == 1
    assert provider.submissions == 1
    assert provider.result_calls == 1
    assert artifacts.writes == 1
    assert len(reservations) == 1
    assert reservations[0][0].startswith("corpus-parse:parse:llamacloud:")
    assert resumed_reconciled[0][1] == 1


def test_usage_unknown_is_not_reported_as_zero():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider(result={"blocks": RESULT["blocks"], "usage": {"pages": "unknown"}})
    parser, _, _ = coordinator(provider, checkpoints, artifacts)

    outcome = parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)

    assert outcome.state == "awaiting_review"
    assert outcome.page_count == 1
    assert outcome.block_count == 1
    assert outcome.usage_kind is None
    assert outcome.usage_units is None
    assert checkpoints.records[ATTEMPT]["parsing_usage_units"] is None


def test_sigterm_stop_keeps_saved_job_for_a_later_run():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider(statuses=["running", "succeeded"])
    terminated = Event()

    def wait_for_status(_):
        terminated.set()

    parser, _, _ = coordinator(
        provider, checkpoints, artifacts,
        stop_requested=terminated.is_set, sleep=wait_for_status,
    )
    interrupted = parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)
    assert interrupted.state == "parsing"
    assert interrupted.remote_job_id == "job-1"
    assert checkpoints.records[ATTEMPT]["remote_job_id"] == "job-1"

    resumed, _, _ = coordinator(provider, checkpoints, artifacts)
    complete = resumed.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)

    assert complete.state == "awaiting_review"
    assert provider.submissions == 1


def test_ambiguous_submit_outcome_is_durably_paused_without_retry():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider(unknown_submit=True)
    parser, _, _ = coordinator(provider, checkpoints, artifacts)
    with pytest.raises(ParsingOutcomeUnknown):
        parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)
    resumed, _, _ = coordinator(provider, checkpoints, artifacts)
    assert resumed.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF).state == "awaiting_budget"
    assert provider.submissions == 1
    assert provider.lookups == 0


def test_failed_allowance_reconciliation_does_not_mark_artifact_review_ready():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider()
    reconcile_calls = []

    def reconcile(key, units):
        reconcile_calls.append((key, units))
        if len(reconcile_calls) == 1:
            raise RuntimeError("allowance store unavailable")

    parser, _, _ = coordinator(provider, checkpoints, artifacts, reconcile=reconcile)
    with pytest.raises(RuntimeError, match="allowance store unavailable"):
        parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)
    assert "parsing_artifact" not in checkpoints.records[ATTEMPT]

    resumed, _, _ = coordinator(provider, checkpoints, artifacts, reconcile=reconcile)
    outcome = resumed.run(attempt_id=ATTEMPT, source=SOURCE, pdf=PDF)

    assert outcome.state == "awaiting_review"
    assert outcome.artifact is not None
    assert len(reconcile_calls) == 2
    assert provider.submissions == 1
    assert provider.result_calls == 2


def test_pinned_source_change_is_rejected_before_provider_call():
    checkpoints, artifacts = Checkpoints(), Artifacts()
    provider = Provider()
    parser, _, _ = coordinator(provider, checkpoints, artifacts)
    with pytest.raises(ParsingError, match="digest"):
        parser.run(attempt_id=ATTEMPT, source=SOURCE, pdf=b"%PDF-1.7 changed")
    assert provider.submissions == 0


@pytest.mark.parametrize(
    "block",
    [
        {"block_id": 1, "physical_page": 1, "kind": "text", "raw_text": "text"},
        {"block_id": "b1", "physical_page": "1", "kind": "text", "raw_text": "text"},
        {"block_id": "b1", "physical_page": 1, "kind": "text", "raw_text": 7},
    ],
)
def test_parser_rejects_provider_block_type_coercion(block):
    with pytest.raises(ParsingError, match="schema"):
        _parse_result({"blocks": [block]}, "12", SOURCE["sha256"], b"raw")


def test_llamacloud_sdk_items_pages_map_to_strict_source_blocks():
    raw = b"provider response bytes"
    payload = {
        "items": {
            "pages": [
                {
                    "page_number": 21,
                    "success": True,
                    "items": [
                        {"type": "text", "md": "Heading and body"},
                        {
                            "type": "list",
                            "md": "- first\n- second",
                            "items": [
                                {"type": "text", "md": "first"},
                                {"type": "text", "md": "second"},
                            ],
                        },
                        {"type": "table", "md": "A | B", "csv": "A,B\n1,2"},
                    ],
                }
            ],
        },
        "usage": {"credits": 15},
    }

    artifact = _parse_result(payload, "12", SOURCE["sha256"], raw)

    assert [block.kind for block in artifact.blocks] == ["text", "list", "table"]
    assert [block.physical_page for block in artifact.blocks] == [21, 21, 21]
    assert artifact.blocks[1].list_items == ("first", "second")
    assert artifact.blocks[2].table_rows == (("A", "B"), ("1", "2"))
    assert artifact.output_sha256 == hashlib.sha256(raw).hexdigest()


def test_llamacloud_rejects_malformed_nested_list_item():
    payload = {
        "items": {
            "pages": [{
                "page_number": 1,
                "items": [{"type": "list", "items": [
                    {"type": "text", "md": "valid"}, {"md": "malformed"},
                ]}],
            }],
        },
    }
    with pytest.raises(ParsingError, match="list items"):
        _parse_result(payload, "12", SOURCE["sha256"], b"raw")


def test_llamacloud_adapter_uses_account_default_tier_and_fixed_provider_route():
    class Parsing:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("Response", (), {"id": "job-1"})()

    provider = object.__new__(LlamaCloudParsingProvider)
    provider.client = type("Client", (), {"parsing": Parsing()})()
    provider.version = "latest"

    assert provider.submit(pdf=PDF, idempotency_key="operation", route="llamacloud") == "job-1"
    assert "tier" not in provider.client.parsing.kwargs
    assert provider.client.parsing.kwargs["upload_file"] == ("source.pdf", PDF, "application/pdf")
    with pytest.raises(ParsingError, match="route"):
        provider.submit(pdf=PDF, idempotency_key="operation", route="other")


def test_parser_dispatcher_starts_only_the_fixed_job():
    calls = []
    dispatcher = CloudRunHostedParsingDispatcher(
        project="project", location="europe-west1", job_name="dmbok-compass-parser",
        execute=lambda: calls.append("run"),
    )

    dispatcher.dispatch()

    assert calls == ["run"]
