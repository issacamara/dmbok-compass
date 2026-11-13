"""Resumable, content-addressed LlamaCloud parsing for staged PDFs.

The provider-facing protocol is intentionally narrow. The worker persists a
remote job reference before polling it and pauses safely if submission outcome
cannot be recovered.
"""

from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from app.contracts import ParsedBlock, ParsingArtifact
from app.telemetry import LoggingTelemetryAdapter, TelemetryAdapter, stage_event


class ParsingError(RuntimeError):
    """A safe, non-provider-specific parsing failure."""


class ParsingOutcomeUnknown(ParsingError):
    """The paid submission outcome cannot be safely reconciled."""


class ParsingAttemptNotFound(ParsingError):
    """The staging attempt requested for hosted parsing does not exist."""


class ParsingDispatchError(ParsingError):
    """The fixed Cloud Run parsing job could not be started."""


class ParsingProvider(Protocol):
    def submit(self, *, pdf: bytes, idempotency_key: str, route: str) -> str: ...

    def status(self, job_id: str) -> Literal["pending", "running", "succeeded", "failed"]: ...

    def result(self, job_id: str) -> tuple[bytes, Any]: ...


class LlamaCloudParsingProvider:
    """Adapter for the pinned `llama-cloud` v2 Parse SDK."""

    def __init__(self, api_key: str, *, version: str = "latest") -> None:
        if not api_key.strip():
            raise ValueError("LlamaCloud credential is required")
        from llama_cloud import LlamaCloud

        self.client = LlamaCloud(api_key=api_key)
        self.version = version

    def submit(self, *, pdf: bytes, idempotency_key: str, route: str) -> str:
        # SDK 2.19.0 has no lookup by idempotency key; this key is retained in
        # our durable operation record, and ambiguous requests are paused.
        del idempotency_key
        if route != "llamacloud":
            raise ParsingError("unsupported hosted parsing route")
        response = self.client.parsing.create(
            tier="cost_effective",
            version=self.version,
            upload_file=("source.pdf", pdf, "application/pdf"),
        )
        job_id = getattr(response, "id", None)
        if not isinstance(job_id, str) or not job_id:
            raise ParsingError("LlamaCloud did not return a parse job ID")
        return job_id

    def status(self, job_id: str) -> Literal["pending", "running", "succeeded", "failed"]:
        response = self.client.parsing.get(job_id)
        job = getattr(response, "job", None)
        status = getattr(job, "status", None)
        if status == "COMPLETED":
            return "succeeded"
        if status in {"FAILED", "CANCELLED"}:
            return "failed"
        if status in {"PENDING", "RUNNING"}:
            return "pending" if status == "PENDING" else "running"
        raise ParsingError("LlamaCloud returned an unsupported parse status")

    def result(self, job_id: str) -> tuple[bytes, Any]:
        response = self.client.parsing.with_raw_response.get(
            job_id, expand=["items", "usage"]
        )
        raw = response.read()
        parsed = response.parse()
        return raw, _sdk_model_dump(parsed)


class ParsingCheckpointStore(Protocol):
    def load(self, attempt_id: str) -> Mapping[str, Any]: ...

    def record_job(self, attempt_id: str, job_id: str, operation_key: str) -> None: ...

    def mark_submission_started(self, attempt_id: str, operation_key: str) -> None: ...

    def record_artifact(self, attempt_id: str, artifact: Mapping[str, Any]) -> None: ...

    def record_reservation(self, attempt_id: str, reservation: Mapping[str, Any]) -> None: ...

    def record_status(
        self, attempt_id: str, state: str, *, error_code: str | None = None,
        page_count: int | None = None, block_count: int | None = None,
        usage_kind: str | None = None, usage_units: int | None = None,
    ) -> None: ...


class ParsingArtifactStore(Protocol):
    def get(self, path: str, generation: int | None = None) -> bytes: ...

    def put_if_absent(self, path: str, content: bytes, content_type: str) -> Mapping[str, Any]: ...


class FirestoreParsingCheckpointStore:
    """Transactional parsing checkpoints attached to the existing staging attempt."""

    def __init__(self, client: Any, *, transaction_runner: Callable[[Any, Callable[[Any], Any]], Any] | None = None) -> None:
        from app.corpus.staging import _run_firestore_transaction

        self.client = client
        self.attempts = client.collection("corpus_staging_attempts")
        self.transaction_runner = transaction_runner or _run_firestore_transaction

    def load(self, attempt_id: str) -> Mapping[str, Any]:
        snapshot = self.attempts.document(attempt_id).get()
        if not snapshot.exists:
            raise ParsingError("staging attempt is unavailable")
        return snapshot.to_dict()

    def enqueue(self, attempt_id: str) -> Mapping[str, Any]:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> Mapping[str, Any]:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingAttemptNotFound("staging attempt is unavailable")
            record = snapshot.to_dict()
            if not isinstance(record.get("source"), dict):
                raise ParsingError("staged source is unavailable")
            if record.get("parsing_artifact"):
                return record
            if record.get("parsing_submission_started") and not record.get("remote_job_id"):
                raise ParsingOutcomeUnknown("previous parser submission requires reconciliation")
            if record.get("parsing_state") not in {None, "queued", "awaiting_budget", "parsing"}:
                raise ParsingError("staging attempt is not eligible for hosted parsing")
            transaction.update(reference, {
                "parsing_state": "queued",
                "parsing_error_code": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            return {**record, "parsing_state": "queued", "parsing_error_code": None}

        return self.transaction_runner(self.client.transaction(), apply)


    def claim_next(self) -> tuple[str, Mapping[str, Any]] | None:
        candidates = self.attempts.where("parsing_state", "==", "queued").limit(20).stream()
        for candidate in candidates:
            reference = self.attempts.document(candidate.id)

            def apply(transaction: Any) -> Mapping[str, Any] | None:
                snapshot = reference.get(transaction=transaction)
                if not snapshot.exists:
                    return None
                record = snapshot.to_dict()
                if record.get("parsing_state") != "queued":
                    return None
                transaction.update(reference, {
                    "parsing_state": "parsing",
                    "parsing_claimed_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                return record

            record = self.transaction_runner(self.client.transaction(), apply)
            if record is not None:
                return candidate.id, record
        return None

    def record_job(self, attempt_id: str, job_id: str, operation_key: str) -> None:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingError("staging attempt is unavailable")
            record = snapshot.to_dict()
            existing = record.get("remote_job_id")
            existing_key = record.get("parsing_operation_key")
            if existing and (existing != job_id or existing_key != operation_key):
                raise ParsingError("staging attempt is already bound to another parser job")
            if not isinstance(record.get("source"), dict):
                raise ParsingError("staging source is unavailable")
            transaction.update(reference, {
                "remote_job_id": job_id,
                "parsing_operation_key": operation_key,
                "parsing_state": "parsing",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        self.transaction_runner(self.client.transaction(), apply)

    def mark_submission_started(self, attempt_id: str, operation_key: str) -> None:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingError("staging attempt is unavailable")
            record = snapshot.to_dict()
            existing_key = record.get("parsing_operation_key")
            if existing_key is not None and existing_key != operation_key:
                raise ParsingError("staging attempt is already bound to another parsing operation")
            if record.get("remote_job_id"):
                return
            if not isinstance(record.get("source"), dict):
                raise ParsingError("staging source is unavailable")
            transaction.update(reference, {
                "parsing_operation_key": operation_key,
                "parsing_submission_started": True,
                "parsing_state": "parsing",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        self.transaction_runner(self.client.transaction(), apply)

    def record_artifact(self, attempt_id: str, artifact: Mapping[str, Any]) -> None:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingError("staging attempt is unavailable")
            record = snapshot.to_dict()
            if record.get("remote_job_id") != artifact.get("remote_job_id"):
                raise ParsingError("parser artifact is not bound to the saved remote job")
            existing = record.get("parsing_artifact")
            if existing and existing != dict(artifact):
                raise ParsingError("staging attempt already has a different parsing artifact")
            transaction.update(reference, {
                "parsing_artifact": dict(artifact),
                "parsing_output_sha256": artifact["sha256"],
                "parsing_state": "awaiting_review",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        self.transaction_runner(self.client.transaction(), apply)

    def record_reservation(self, attempt_id: str, reservation: Mapping[str, Any]) -> None:
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingError("staging attempt is unavailable")
            record = snapshot.to_dict()
            existing = record.get("allowance_reservation")
            if existing and existing != dict(reservation):
                raise ParsingError("staging attempt already has a different allowance reservation")
            transaction.update(reference, {
                "allowance_reservation": dict(reservation),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        self.transaction_runner(self.client.transaction(), apply)

    def record_status(
        self, attempt_id: str, state: str, *, error_code: str | None = None,
        page_count: int | None = None, block_count: int | None = None,
        usage_kind: str | None = None, usage_units: int | None = None,
    ) -> None:
        if state not in {"awaiting_budget", "parsing", "awaiting_review", "failed"}:
            raise ValueError("invalid hosted parsing state")
        if page_count is not None and page_count < 0 or block_count is not None and block_count < 0:
            raise ValueError("parsing counts must be non-negative")
        if usage_units is not None and (type(usage_units) is not int or usage_units < 0):
            raise ValueError("parsing usage must be a non-negative integer")
        if (usage_units is None) != (usage_kind is None):
            raise ValueError("parsing usage value and kind must be provided together")
        reference = self.attempts.document(attempt_id)

        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ParsingError("staging attempt is unavailable")
            payload = {
                "parsing_state": state,
                "parsing_error_code": error_code,
                "parsing_page_count": page_count,
                "parsing_block_count": block_count,
                "parsing_usage_kind": usage_kind,
                "parsing_usage_units": usage_units,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            transaction.update(reference, payload)

        self.transaction_runner(self.client.transaction(), apply)

class CloudRunHostedParsingDispatcher:
    """Start the fixed parser job without caller-controlled overrides."""

    def __init__(
        self, *, project: str, location: str, job_name: str,
        execute: Callable[[], None] | None = None,
    ) -> None:
        self.project = _required(project, "project")
        self.location = _required(location, "location")
        self.job_name = _required(job_name, "job_name")
        self.execute = execute or self._execute_cloud_run

    def dispatch(self) -> None:
        self.execute()

    def _execute_cloud_run(self) -> None:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        response = AuthorizedSession(credentials).post(
            "https://run.googleapis.com/v2/projects/"
            f"{self.project}/locations/{self.location}/jobs/{self.job_name}:run",
            timeout=15,
        )
        if not response.ok:
            raise ParsingDispatchError("hosted parsing job could not be dispatched")


def _required(value: str, name: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


@dataclass(frozen=True)
class ParsingRun:
    artifact: ParsingArtifact | None
    state: Literal["awaiting_budget", "parsing", "awaiting_review", "failed"]
    remote_job_id: str | None = None
    page_count: int | None = None
    block_count: int | None = None
    usage_kind: str | None = None
    usage_units: int | None = None


class HostedParsingCoordinator:
    """Drive one attempt to a validated artifact without duplicate paid work."""

    def __init__(
        self,
        provider: ParsingProvider,
        checkpoints: ParsingCheckpointStore,
        artifacts: ParsingArtifactStore,
        *,
        bucket_name: str,
        route: str,
        reserve_allowance: Callable[[str, int], Any],
        reconcile_allowance: Callable[[str, int | None], None],
        max_units: int,
        mode: str = "parse",
        deadline_seconds: float = 240,
        poll_interval_seconds: float = 2,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        stop_requested: Callable[[], bool] = lambda: False,
        telemetry: TelemetryAdapter | None = None,
    ) -> None:
        if max_units < 1 or deadline_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("invalid parsing bounds")
        if route != "llamacloud":
            raise ValueError("unsupported hosted parsing route")
        if mode != "parse":
            raise ValueError("unsupported hosted parsing mode")
        self.provider = provider
        self.checkpoints = checkpoints
        self.artifacts = artifacts
        self.bucket_name = bucket_name
        self.route = route
        self.mode = mode
        self.reserve_allowance = reserve_allowance
        self.reconcile_allowance = reconcile_allowance
        self.max_units = max_units
        self.deadline_seconds = deadline_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.monotonic = monotonic
        self.sleep = sleep
        self.stop_requested = stop_requested
        self.telemetry = telemetry or LoggingTelemetryAdapter()

    def run(
        self, *, attempt_id: str, source: Mapping[str, Any], pdf: bytes,
        physical_page_numbers: list[int] | None = None,
    ) -> ParsingRun:
        if not attempt_id.strip() or not _valid_source(source, self.bucket_name, attempt_id):
            raise ParsingError("staged source binding is invalid")
        if not pdf.startswith(b"%PDF-"):
            raise ParsingError("staged source is not a PDF")
        source_digest = hashlib.sha256(pdf).hexdigest()
        if source_digest != source.get("sha256"):
            raise ParsingError("staged source digest does not match its pinned identity")

        checkpoint = self.checkpoints.load(attempt_id)
        saved = _saved_artifact(
            checkpoint,
            artifacts=self.artifacts,
            source_generation=str(source["generation"]),
            source_digest=source_digest,
        )
        if saved is not None:
            return ParsingRun(
                saved, "awaiting_review", str(checkpoint.get("remote_job_id") or "") or None,
                len({block.physical_page for block in saved.blocks}), len(saved.blocks),
                checkpoint.get("parsing_usage_kind"), checkpoint.get("parsing_usage_units"),
            )

        if physical_page_numbers is not None:
            pdf = _extract_selected_pdf_pages(pdf, physical_page_numbers)
            selected_key = json.dumps(physical_page_numbers, separators=(",", ":"))
        else:
            selected_key = "all"
        operation_key = f"corpus-parse:{self.mode}:{self.route}:{attempt_id}:{source_digest}"
        if physical_page_numbers is not None:
            operation_key = f"{operation_key}:{selected_key}"
        existing_operation_key = checkpoint.get("parsing_operation_key")
        if existing_operation_key is not None and existing_operation_key != operation_key:
            raise ParsingError("staging attempt parser settings changed")
        if checkpoint.get("parsing_error_code") == "submission_unresolved":
            return ParsingRun(None, "awaiting_budget")
        reservation = checkpoint.get("allowance_reservation")
        if not isinstance(reservation, Mapping):
            reserved = self.reserve_allowance(operation_key, self.max_units)
            if not reserved:
                self.checkpoints.record_status(attempt_id, "awaiting_budget")
                return ParsingRun(None, "awaiting_budget")
            reservation_payload = reserved.model_dump(mode="json") if hasattr(reserved, "model_dump") else reserved
            if isinstance(reservation_payload, Mapping):
                self.checkpoints.record_reservation(attempt_id, reservation_payload)

        remote_job_id = checkpoint.get("remote_job_id")
        if remote_job_id is not None and not isinstance(remote_job_id, str):
            raise ParsingError("saved parser job identity is invalid")
        if not remote_job_id:
            if checkpoint.get("parsing_submission_started"):
                self.checkpoints.record_status(attempt_id, "awaiting_budget", error_code="submission_unresolved")
                return ParsingRun(None, "awaiting_budget")
            self.checkpoints.mark_submission_started(attempt_id, operation_key)
            try:
                remote_job_id = self.provider.submit(
                    pdf=pdf, idempotency_key=operation_key, route=self.route
                )
            except Exception:
                self.checkpoints.record_status(attempt_id, "awaiting_budget", error_code="submission_unresolved")
                raise ParsingOutcomeUnknown("parser submission outcome is unresolved") from None
            if not isinstance(remote_job_id, str) or not remote_job_id.strip():
                self.checkpoints.record_status(attempt_id, "awaiting_budget", error_code="submission_unresolved")
                return ParsingRun(None, "awaiting_budget")
            self.checkpoints.record_job(attempt_id, remote_job_id, operation_key)

        self.checkpoints.record_status(attempt_id, "parsing")
        deadline = self.monotonic() + self.deadline_seconds
        while True:
            if self.stop_requested():
                return ParsingRun(None, "parsing", remote_job_id)
            try:
                status = self.provider.status(remote_job_id)
            except Exception:
                self.checkpoints.record_status(attempt_id, "parsing", error_code="parser_status_unavailable")
                return ParsingRun(None, "parsing", remote_job_id)
            if status == "failed":
                self.reconcile_allowance(operation_key, None)
                self.checkpoints.record_status(attempt_id, "failed", error_code="parser_failed")
                return ParsingRun(None, "failed", remote_job_id)
            if status == "succeeded":
                return self._persist_result(
                    attempt_id, source, source_digest, remote_job_id, operation_key,
                    physical_page_numbers,
                )
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return ParsingRun(None, "parsing", remote_job_id)
            self.sleep(min(self.poll_interval_seconds, remaining))

    def _persist_result(
        self, attempt_id: str, source: Mapping[str, Any], source_digest: str,
        job_id: str, operation_key: str, physical_page_numbers: list[int] | None = None,
    ) -> ParsingRun:
        try:
            raw_result, payload = self.provider.result(job_id)
            artifact = _parse_result(
                payload, str(source["generation"]), source_digest, raw_result,
                physical_page_numbers=physical_page_numbers,
            )
        except (ParsingError, ValidationError):
            self.reconcile_allowance(operation_key, None)
            self.checkpoints.record_status(attempt_id, "failed", error_code="invalid_parser_output")
            return ParsingRun(None, "failed", job_id)
        except Exception as error:
            self.checkpoints.record_status(attempt_id, "parsing", error_code="parser_result_unavailable")
            self.telemetry.emit(stage_event(
                "parsing", "failed", error=error, http_status_code=_http_status_code(error)
            ))
            return ParsingRun(None, "parsing", job_id)
        usage_kind, usage_units = _usage_detail(payload)
        self.reconcile_allowance(operation_key, usage_units)
        if usage_units is not None and usage_units > self.max_units:
            self.checkpoints.record_status(attempt_id, "failed", error_code="parser_usage_exceeded")
            return ParsingRun(None, "failed", job_id, usage_kind=usage_kind, usage_units=usage_units)
        serialized = artifact.model_dump_json().encode("utf-8")
        artifact_digest = hashlib.sha256(serialized).hexdigest()
        digest = artifact.output_sha256
        path = f"staging/{attempt_id}/parsing/{digest}.json"
        try:
            stored = self.artifacts.put_if_absent(path, serialized, "application/json")
        except Exception:
            self.checkpoints.record_status(attempt_id, "parsing", error_code="artifact_persistence_failed")
            return ParsingRun(None, "parsing", job_id)
        identity = {
            "object_name": path,
            "sha256": digest,
            "source_generation": str(source["generation"]),
            "source_sha256": source_digest,
            "artifact_sha256": artifact_digest,
            "artifact_generation": stored.get("generation"),
            "remote_job_id": job_id,
        }
        self.checkpoints.record_artifact(attempt_id, identity)
        page_count = len({block.physical_page for block in artifact.blocks})
        block_count = len(artifact.blocks)
        self.checkpoints.record_status(
            attempt_id, "awaiting_review", page_count=page_count, block_count=block_count,
            usage_kind=usage_kind, usage_units=usage_units,
        )
        return ParsingRun(artifact, "awaiting_review", job_id, page_count, block_count, usage_kind, usage_units)


def _valid_source(source: Mapping[str, Any], bucket_name: str, attempt_id: str) -> bool:
    return (
        source.get("bucket") == bucket_name
        and source.get("object_name") == f"staging/{attempt_id}/source.pdf"
        and isinstance(source.get("generation"), (str, int))
        and str(source.get("generation", "")).isdigit()
        and isinstance(source.get("sha256"), str)
        and len(str(source.get("sha256"))) == 64
    )


def _saved_artifact(
    checkpoint: Mapping[str, Any],
    *, artifacts: ParsingArtifactStore, source_generation: str, source_digest: str
) -> ParsingArtifact | None:
    identity = checkpoint.get("parsing_artifact")
    if not isinstance(identity, Mapping) or identity.get("source_generation") != source_generation or identity.get("source_sha256") != source_digest:
        return None
    try:
        raw_bytes = artifacts.get(
            str(identity["object_name"]),
            int(identity["artifact_generation"]) if identity.get("artifact_generation") is not None else None,
        )
        if hashlib.sha256(raw_bytes).hexdigest() != identity.get("artifact_sha256"):
            raise ParsingError("saved parsing artifact content does not match")
        artifact = ParsingArtifact.model_validate_json(raw_bytes)
    except (KeyError, TypeError, ValueError, ValidationError):
        raise ParsingError("saved parsing artifact is invalid") from None
    if artifact.output_sha256 != identity.get("sha256"):
        raise ParsingError("saved parsing artifact digest does not match")
    return artifact


def _extract_selected_pdf_pages(pdf: bytes, physical_page_numbers: list[int]) -> bytes:
    if (
        not physical_page_numbers or len(physical_page_numbers) > 5
        or physical_page_numbers != sorted(set(physical_page_numbers))
        or any(type(page) is not int or page < 1 for page in physical_page_numbers)
    ):
        raise ParsingError("physical page selection is invalid")
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf))
        if physical_page_numbers[-1] > len(reader.pages):
            raise ParsingError("physical page selection is outside the source PDF")
        writer = PdfWriter()
        for page_number in physical_page_numbers:
            writer.add_page(reader.pages[page_number - 1])
        output = io.BytesIO()
        writer.write(output)
        subset = output.getvalue()
        if len(PdfReader(io.BytesIO(subset)).pages) != len(physical_page_numbers):
            raise ParsingError("selected PDF page count could not be verified")
        return subset
    except ParsingError:
        raise
    except Exception:
        raise ParsingError("selected PDF pages could not be extracted") from None


def _parse_result(
    payload: Any, source_generation: str, source_digest: str, raw_result: bytes,
    *, physical_page_numbers: list[int] | None = None,
) -> ParsingArtifact:
    if not isinstance(payload, Mapping):
        raise ParsingError("parser result schema is invalid")
    raw_blocks = payload.get("blocks")
    if raw_blocks is None:
        raw_blocks = _llama_cloud_blocks(payload)
    if not isinstance(raw_blocks, list):
        raise ParsingError("parser result schema is invalid")
    blocks: list[ParsedBlock] = []
    try:
        for order, value in enumerate(raw_blocks):
            if not isinstance(value, Mapping):
                raise ParsingError("parser block schema is invalid")
            block_id = value.get("block_id")
            physical_page = value.get("physical_page")
            if physical_page_numbers is not None:
                if type(physical_page) is not int or not 1 <= physical_page <= len(physical_page_numbers):
                    raise ParsingError("parser returned a page outside the selected PDF")
                physical_page = physical_page_numbers[physical_page - 1]
            printed_page_label = value.get("printed_page_label")
            kind = value.get("kind")
            raw_text = value.get("raw_text")
            table_rows = value.get("table_rows", ())
            list_items = value.get("list_items", ())
            if (
                not isinstance(block_id, str)
                or type(physical_page) is not int
                or (printed_page_label is not None and not isinstance(printed_page_label, str))
                or kind not in {"text", "table", "list"}
                or not isinstance(raw_text, str)
                or not isinstance(table_rows, (list, tuple))
                or any(
                    not isinstance(row, (list, tuple))
                    or any(not isinstance(cell, str) for cell in row)
                    for row in table_rows
                )
                or not isinstance(list_items, (list, tuple))
                or any(not isinstance(item, str) for item in list_items)
            ):
                raise ParsingError("parser block schema is invalid")
            blocks.append(ParsedBlock(
                block_id=block_id,
                physical_page=physical_page,
                printed_page_label=printed_page_label,
                reading_order=order,
                kind=kind,
                raw_text=raw_text,
                table_rows=tuple(tuple(row) for row in table_rows),
                list_items=tuple(list_items),
            ))
        if not raw_result:
            raise ParsingError("parser returned an empty result")
        provisional = {
            "schema_version": "parsing-artifact-v1",
            "source_generation": source_generation,
            "source_sha256": source_digest,
            "output_sha256": hashlib.sha256(raw_result).hexdigest(),
            "blocks": [block.model_dump(mode="json") for block in blocks],
        }
        return ParsingArtifact.model_validate(provisional)
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise ParsingError("parser result schema is invalid") from error


def _llama_cloud_blocks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map generated SDK `Items.pages` records using their documented fields."""
    items = payload.get("items")
    if not isinstance(items, Mapping) or not isinstance(items.get("pages"), list):
        raise ParsingError("LlamaCloud items payload is unavailable")
    blocks: list[dict[str, Any]] = []
    for page in items["pages"]:
        if not isinstance(page, Mapping) or page.get("success") is False:
            raise ParsingError("LlamaCloud returned an unsuccessful page")
        page_number = page.get("page_number")
        page_items = page.get("items")
        if type(page_number) is not int or page_number < 1 or not isinstance(page_items, list):
            raise ParsingError("LlamaCloud page schema is invalid")
        for item in page_items:
            if not isinstance(item, Mapping):
                raise ParsingError("LlamaCloud structured item schema is invalid")
            item_type = item.get("type")
            markdown = item.get("md")
            block_id = f"page-{page_number}-item-{len(blocks)}"
            if item_type == "table":
                csv_text = item.get("csv")
                if not isinstance(csv_text, str) or not csv_text.strip():
                    raise ParsingError("LlamaCloud table content is invalid")
                rows = tuple(tuple(row) for row in csv.reader(io.StringIO(csv_text)))
                if not rows or not all(rows):
                    raise ParsingError("LlamaCloud table rows are invalid")
                blocks.append({
                    "block_id": block_id,
                    "physical_page": page_number,
                    "kind": "table",
                    "raw_text": markdown if isinstance(markdown, str) and markdown else csv_text,
                    "table_rows": rows,
                })
                continue
            if item_type == "list":
                nested = item.get("items")
                if not isinstance(nested, list):
                    raise ParsingError("LlamaCloud list content is invalid")
                if not nested or any(
                    not isinstance(child, Mapping)
                    or child.get("type") != "text"
                    or not isinstance(child.get("md"), str)
                    or not child["md"].strip()
                    for child in nested
                ):
                    raise ParsingError("LlamaCloud list items are invalid")
                list_items = tuple(child["md"] for child in nested)
                blocks.append({
                    "block_id": block_id,
                    "physical_page": page_number,
                    "kind": "list",
                    "raw_text": markdown if isinstance(markdown, str) and markdown else "\n".join(list_items),
                    "list_items": list_items,
                })
                continue
            if item_type in {"text", "heading", "code", "link", "image"} and isinstance(markdown, str) and markdown.strip():
                blocks.append({
                    "block_id": block_id,
                    "physical_page": page_number,
                    "kind": "text",
                    "raw_text": markdown,
                })
                continue
            raise ParsingError("LlamaCloud item type or content is unsupported")
    return blocks


def _http_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    for value in (getattr(error, "status_code", None), getattr(response, "status_code", None)):
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


def _sdk_model_dump(response: Any) -> Mapping[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif hasattr(response, "dict"):
        value = response.dict()
    elif isinstance(response, Mapping):
        value = response
    else:
        raise ParsingError("LlamaCloud result model is invalid")
    if not isinstance(value, Mapping):
        raise ParsingError("LlamaCloud result model is invalid")
    return value


def _usage_units(payload: Any) -> int | None:
    return _usage_detail(payload)[1]


def _usage_detail(payload: Any) -> tuple[str | None, int | None]:
    if not isinstance(payload, Mapping):
        return None, None
    job = payload.get("job")
    for usage in (payload.get("usage"), job.get("usage") if isinstance(job, Mapping) else None):
        if isinstance(usage, Mapping):
            for key in ("pages", "units", "credits"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    return key, value
                if key == "credits" and type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    return key, math.ceil(value)
    return None, None
