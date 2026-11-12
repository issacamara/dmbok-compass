"""Fixed Cloud Run entry point for resumable hosted PDF parsing."""

from __future__ import annotations

import hashlib
import json
import os
import signal
from threading import Event
from typing import Any

from google.api_core.exceptions import PreconditionFailed
from google.cloud import firestore, storage

from app.ingestion.hosted_parsing import (
    FirestoreParsingCheckpointStore, HostedParsingCoordinator,
    LlamaCloudParsingProvider, ParsingError, ParsingOutcomeUnknown,
)
from app.telemetry import LoggingTelemetryAdapter, stage_event
from app.provider_allowance import ProviderAccountCapabilities
from app.provider_allowance_firestore import FirestoreProviderAllowanceStore
from app.settings import ParsingConfiguration


class _GcsArtifactStore:
    def __init__(self, bucket: Any) -> None:
        self.bucket = bucket

    def get(self, path: str, generation: int | None = None) -> bytes:
        blob = self.bucket.blob(path, generation=generation)
        return blob.download_as_bytes(timeout=60)

    def put_if_absent(self, path: str, content: bytes, content_type: str) -> dict[str, int]:
        blob = self.bucket.blob(path)
        try:
            blob.upload_from_string(
                content, content_type=content_type, if_generation_match=0, timeout=60
            )
        except PreconditionFailed:
            existing = self.bucket.blob(path)
            existing.reload(timeout=30)
            generation = int(existing.generation)
            if self.get(path, generation) != content:
                raise ParsingError("content-addressed parsing artifact conflicts") from None
            return {"generation": generation}
        return {"generation": int(blob.generation)}


def main() -> None:
    terminated = Event()
    old_handler = signal.signal(signal.SIGTERM, lambda *_: terminated.set())
    try:
        _run(terminated)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def _run(terminated: Event) -> None:
    configuration = ParsingConfiguration.from_environment()
    project = _required("GOOGLE_CLOUD_PROJECT")
    bucket_name = _required("CORPUS_BUCKET_NAME")
    database = firestore.Client(project=project)
    checkpoint_store = FirestoreParsingCheckpointStore(database)
    queued = checkpoint_store.claim_next()
    if queued is None:
        print(json.dumps({"event": "hosted_parsing_idle"}))
        return
    attempt_id, record = queued
    telemetry = LoggingTelemetryAdapter()
    telemetry.emit(stage_event("parsing", "resumed" if record.get("remote_job_id") else "started"))
    source = record.get("source")
    if not isinstance(source, dict) or source.get("bucket") != bucket_name:
        checkpoint_store.record_status(attempt_id, "failed", error_code="source_binding_invalid")
        telemetry.emit(stage_event("parsing", "failed", error=ParsingError("invalid source binding")))
        return

    storage_client = storage.Client(project=project)
    try:
        source_blob = storage_client.bucket(bucket_name).blob(
            str(source["object_name"]), generation=int(source["generation"])
        )
        pdf = source_blob.download_as_bytes(timeout=60)
    except Exception:
        checkpoint_store.record_status(attempt_id, "failed", error_code="source_unavailable")
        telemetry.emit(stage_event("parsing", "failed", error=ParsingError("source unavailable")))
        return
    if hashlib.sha256(pdf).hexdigest() != source.get("sha256"):
        checkpoint_store.record_status(attempt_id, "failed", error_code="source_digest_mismatch")
        telemetry.emit(stage_event("parsing", "failed", error=ParsingError("source digest mismatch")))
        return

    allowances = FirestoreProviderAllowanceStore(database)
    capabilities = ProviderAccountCapabilities(
        provider_id="llamacloud",
        account_id=configuration.allowance_policy,
        kind="parsing",
        max_units_per_operation=configuration.allowance_max_units,
        max_retries=0,
    )
    provider = LlamaCloudParsingProvider(
        os.environ["LLAMACLOUD_API_KEY"], version="latest"
    )
    coordinator = HostedParsingCoordinator(
        provider,
        checkpoint_store,
        _GcsArtifactStore(storage_client.bucket(bucket_name)),
        bucket_name=bucket_name,
        route=configuration.route,
        mode=configuration.mode,
        reserve_allowance=lambda key, units: _reserve(allowances, capabilities, key, units),
        reconcile_allowance=lambda key, units: allowances.reconcile_usage(
            capabilities, key, units
        ),
        max_units=configuration.allowance_max_units,
        deadline_seconds=840,
        poll_interval_seconds=3,
        stop_requested=terminated.is_set,
    )
    try:
        outcome = coordinator.run(
            attempt_id=attempt_id, source=source, pdf=pdf,
            physical_page_numbers=record.get("physical_page_numbers"),
        )
    except ParsingOutcomeUnknown:
        telemetry.emit(stage_event("parsing", "awaiting_budget"))
        return
    except Exception as exc:
        telemetry.emit(stage_event("parsing", "failed", error=exc))
        raise RuntimeError("hosted parsing stage failed") from None
    if outcome.state == "awaiting_budget":
        event = stage_event("parsing", "awaiting_budget")
    elif outcome.state == "failed":
        event = stage_event("parsing", "failed")
    elif outcome.state == "awaiting_review":
        event = stage_event(
            "parsing", "succeeded",
            completed_count=outcome.block_count, total_count=outcome.block_count,
            usage_kind=outcome.usage_kind, usage_units=outcome.usage_units,
        )
    else:
        event = stage_event("parsing", "resumed" if terminated.is_set() else "started")
    telemetry.emit(event)
    print(json.dumps({
        "event": "hosted_parsing_finished",
        "state": outcome.state,
        "has_remote_job": outcome.remote_job_id is not None,
        "has_artifact": outcome.artifact is not None,
        "termination_requested": terminated.is_set(),
    }))


def _reserve(allowances: FirestoreProviderAllowanceStore, capabilities: ProviderAccountCapabilities, key: str, units: int):
    result = allowances.stage_reservation(capabilities, key, units, retries=0)
    return result.reservation if result.status == "staged" else None


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required parser setting: {name}")
    return value


if __name__ == "__main__":
    main()
