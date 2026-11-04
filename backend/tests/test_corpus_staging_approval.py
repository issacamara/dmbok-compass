from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

import pytest

from app.contracts import ApiError, CorpusStagingAttemptStatus, CorpusStagingPreviewEntry
from app.corpus.staging import (
    CloudRunCorpusIngestionDispatcher,
    CorpusStagingApprovalService,
    StagingAttemptConflict,
    StagingDispatchError,
)


ATTEMPT = "approval-test"
BUCKET = "private-bucket"
PDF = b"%PDF-1.7\nDMBOK\n%%EOF"
TEMPLATE = json.dumps({
    "schema_version": "structure-template-v1", "document_title": "DMBOK",
    "instructions": ["Map the sections."],
    "entries": [{"section_path": ["Chapter 1"], "page_start": 1, "page_end": 1}],
}).encode()


def _artifact(kind: str, generation: int, payload: bytes) -> dict[str, Any]:
    name = f"staging/{ATTEMPT}/source.pdf" if kind == "source" else f"staging/{ATTEMPT}/structure.json"
    return {"bucket": BUCKET, "object_name": name, "generation": generation,
            "sha256": sha256(payload).hexdigest(),
            "content_type": "application/pdf" if kind == "source" else "application/json"}


class Store:
    def __init__(self, *, matched: bool = True) -> None:
        self.record = {
            "attempt_id": ATTEMPT, "revision": 4, "state": "awaiting_review",
            "completed_stage": "review", "preview": [CorpusStagingPreviewEntry(
                entry_index=0, section_path=("Chapter 1",), page_start=1, page_end=1,
                match_status="matched" if matched else "unmatched",
                correction_codes=[] if matched else ["unmatched_passage"],
            ).model_dump(mode="json")],
            "correction_codes": [] if matched else ["unmatched_passage"],
            "source": _artifact("source", 17, PDF), "structure": _artifact("structure", 23, TEMPLATE),
            "preview_binding": {"preview_revision": 4, "source_generation": 17, "structure_generation": 23},
        }

    def get(self, *, attempt_id: str) -> dict[str, Any] | None:
        return deepcopy(self.record) if attempt_id == ATTEMPT else None

    def claim_approval(self, *, attempt_id: str, expected_revision: int, approval: dict[str, Any]):
        status = self.status()
        if status.state in {"queued", "running", "staged"}:
            return status, False
        retry = status.state == "failed" and bool(self.record.get("dispatch_retryable"))
        if status.revision != expected_revision or status.state != "awaiting_review" and not retry:
            raise StagingAttemptConflict("stale")
        if self.record["source"] != approval["_source_binding"] or self.record["structure"] != approval["_structure_binding"]:
            raise StagingAttemptConflict("pair changed")
        status = CorpusStagingAttemptStatus(attempt_id=ATTEMPT, revision=status.revision + 1,
                                             state="queued", completed_stage="approval")
        self.record.update(status.model_dump(mode="json"), dispatch_claimed=True, dispatch_retryable=False,
                           approval={k: v for k, v in approval.items() if not k.startswith("_")},
                           approved_map_artifact=approval["_map_artifact"],
                           approval_artifact=approval["_approval_artifact"])
        return status, True

    def fail_dispatch(self, *, attempt_id: str, expected_revision: int, retryable: bool):
        status = self.status()
        if status.revision != expected_revision or status.state != "queued":
            return status
        error = ApiError(code="job_failed", message="Dispatch failed safely.", retryable=retryable, status=503)
        status = CorpusStagingAttemptStatus(attempt_id=ATTEMPT, revision=status.revision + 1,
                                             state="failed", completed_stage="approval", error=error)
        self.record.update(status.model_dump(mode="json"), dispatch_retryable=retryable, dispatch_claimed=not retryable)
        return status

    def status(self):
        return CorpusStagingAttemptStatus.model_validate({k: v for k, v in self.record.items()
                                                           if k in CorpusStagingAttemptStatus.model_fields})


class Blob:
    def __init__(self, bucket: "Bucket", name: str, generation: int | None = None) -> None:
        self.bucket, self.name, self.generation = bucket, name, generation

    def upload_from_string(self, payload: bytes, *, content_type: str, if_generation_match: int, timeout: int) -> None:
        if self.name in self.bucket.objects:
            raise RuntimeError("already exists")
        self.bucket.objects[self.name] = payload
        self.bucket.generations[self.name] = 100 + len(self.bucket.generations)
        self.generation = self.bucket.generations[self.name]

    def reload(self) -> None:
        self.generation = self.bucket.generations[self.name]

    def download_as_bytes(self, *, timeout: int) -> bytes:
        return self.bucket.objects[self.name]


class Bucket:
    def __init__(self) -> None:
        self.objects = {f"staging/{ATTEMPT}/source.pdf": PDF, f"staging/{ATTEMPT}/structure.json": TEMPLATE}
        self.generations = {f"staging/{ATTEMPT}/source.pdf": 17, f"staging/{ATTEMPT}/structure.json": 23}

    def blob(self, name: str, *, generation: int | None = None) -> Blob:
        return Blob(self, name, generation)


class Dispatcher:
    def __init__(self, failure: bool | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def dispatch(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.failure is not None:
            raise StagingDispatchError(retryable=self.failure)


def service(store: Store, dispatcher: Dispatcher, bucket: Bucket | None = None) -> CorpusStagingApprovalService:
    return CorpusStagingApprovalService(store, bucket or Bucket(), BUCKET, dispatcher)


def test_approval_dispatches_exact_pair_once_and_duplicate_returns_queued() -> None:
    store, dispatcher = Store(), Dispatcher()
    approval = service(store, dispatcher)
    queued = approval.approve(attempt_id=ATTEMPT, expected_revision=4, actor_id="admin-1")
    duplicate = approval.approve(attempt_id=ATTEMPT, expected_revision=4, actor_id="admin-1")

    assert queued.state == duplicate.state == "queued"
    assert queued.revision == duplicate.revision == 5
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0]["source"]["generation"] == 17
    assert dispatcher.calls[0]["map_artifact"]["generation"] > 0
    assert dispatcher.calls[0]["approval_artifact"]["generation"] > 0
    assert store.record["source"]["generation"] == 17


def test_definite_rejection_is_retryable_for_same_revision_lineage() -> None:
    store, rejected, bucket = Store(), Dispatcher(failure=True), Bucket()
    failed = service(store, rejected, bucket).approve(attempt_id=ATTEMPT, expected_revision=4, actor_id="admin-1")
    assert failed.state == "failed" and failed.error and failed.error.retryable
    assert store.record["dispatch_claimed"] is False

    accepted = Dispatcher()
    queued = service(store, accepted, bucket).approve(attempt_id=ATTEMPT, expected_revision=failed.revision, actor_id="admin-1")
    assert queued.state == "queued"
    assert len(accepted.calls) == 1
    assert store.record["source"]["generation"] == 17


def test_ambiguous_dispatch_failure_cannot_be_retried() -> None:
    store, dispatcher = Store(), Dispatcher(failure=False)
    failed = service(store, dispatcher).approve(attempt_id=ATTEMPT, expected_revision=4, actor_id="admin-1")
    assert failed.state == "failed" and failed.error and not failed.error.retryable
    assert store.record["dispatch_claimed"] is True
    with pytest.raises(StagingAttemptConflict):
        service(store, Dispatcher()).approve(attempt_id=ATTEMPT, expected_revision=failed.revision, actor_id="admin-1")


def test_stale_or_unmatched_preview_never_dispatches() -> None:
    store, dispatcher = Store(), Dispatcher()
    with pytest.raises(StagingAttemptConflict):
        service(store, dispatcher).approve(attempt_id=ATTEMPT, expected_revision=3, actor_id="admin-1")
    assert dispatcher.calls == []

    store = Store(matched=False)
    with pytest.raises(StagingAttemptConflict):
        service(store, dispatcher).approve(attempt_id=ATTEMPT, expected_revision=4, actor_id="admin-1")
    assert dispatcher.calls == []


def test_cloud_run_dispatch_uses_only_server_derived_pinned_variables() -> None:
    calls: list[dict[str, str]] = []
    dispatcher = CloudRunCorpusIngestionDispatcher(project="p", location="l", job_name="j", execute=calls.append)
    dispatcher.dispatch(
        attempt_id=ATTEMPT,
        source={"bucket": BUCKET, "object_name": "staging/source.pdf", "generation": 17},
        map_artifact={"bucket": BUCKET, "object_name": "staging/map.json", "generation": 18, "sha256": "a" * 64},
        approval_artifact={"bucket": BUCKET, "object_name": "staging/approval.json", "generation": 19, "sha256": "b" * 64},
    )
    assert calls == [{
        "CORPUS_BUCKET_NAME": BUCKET, "CORPUS_OBJECT_NAME": "staging/source.pdf",
        "CORPUS_SOURCE_GENERATION": "17", "CORPUS_STRUCTURE_MAP_URI": f"gs://{BUCKET}/staging/map.json#18",
        "CORPUS_STRUCTURE_MAP_SHA256": "a" * 64,
        "CORPUS_STRUCTURE_MAP_APPROVAL_URI": f"gs://{BUCKET}/staging/approval.json#19",
        "CORPUS_STRUCTURE_MAP_APPROVAL_SHA256": "b" * 64, "CORPUS_STAGING_ATTEMPT_ID": ATTEMPT,
    }]


def test_firestore_approval_claim_is_revision_bound_idempotent_and_retryable_after_rejection() -> None:
    from test_evaluation_generations import FakeFirestore, run_fake_transaction
    from app.corpus.staging import FirestoreCorpusStagingAttemptStore

    database = FakeFirestore()
    store = FirestoreCorpusStagingAttemptStore(database, transaction_runner=run_fake_transaction)
    store.begin(attempt_id=ATTEMPT, actor_id="uploader", source_object_name=f"staging/{ATTEMPT}/source.pdf",
                map_object_name=f"staging/{ATTEMPT}/structure.json")
    source, structure = _artifact("source", 17, PDF), _artifact("structure", 23, TEMPLATE)
    store.attach_artifact(attempt_id=ATTEMPT, artifact_kind="source", artifact=source)
    store.attach_artifact(attempt_id=ATTEMPT, artifact_kind="structure", artifact=structure)
    preview = CorpusStagingPreviewEntry(entry_index=0, section_path=("Chapter 1",), page_start=1, page_end=1,
                                        match_status="matched")
    store.complete_validation(attempt_id=ATTEMPT, expected_revision=3,
                              status=CorpusStagingAttemptStatus(attempt_id=ATTEMPT, revision=3,
                                  state="awaiting_review", completed_stage="review", preview=[preview]),
                              source_generation=17, structure_generation=23)
    map_artifact = {"bucket": BUCKET, "object_name": f"staging/{ATTEMPT}/approved-structure-map.json",
                    "generation": 24, "sha256": "a" * 64, "content_type": "application/json"}
    approval_artifact = {"bucket": BUCKET, "object_name": f"staging/{ATTEMPT}/approval.json",
                         "generation": 25, "sha256": "b" * 64, "content_type": "application/json"}
    approval = {"schema_version": "corpus-owner-approval-v1", "decision": "approved",
                "source": {"uri": f"gs://{BUCKET}/staging/{ATTEMPT}/source.pdf#17", "generation": 17,
                           "sha256": source["sha256"]},
                "_source_binding": source, "_structure_binding": structure,
                "_map_artifact": map_artifact, "_approval_artifact": approval_artifact}

    queued, dispatch = store.claim_approval(attempt_id=ATTEMPT, expected_revision=4, approval=approval)
    duplicate, duplicate_dispatch = store.claim_approval(attempt_id=ATTEMPT, expected_revision=4, approval=approval)
    assert queued.state == duplicate.state == "queued"
    assert queued.revision == duplicate.revision == 5
    assert dispatch and not duplicate_dispatch
    failed = store.fail_dispatch(attempt_id=ATTEMPT, expected_revision=5, retryable=True)
    assert failed.state == "failed" and failed.error and failed.error.retryable
    retried, should_dispatch = store.claim_approval(attempt_id=ATTEMPT, expected_revision=failed.revision, approval=approval)
    assert retried.state == "queued" and should_dispatch
    assert database.data[f"corpus_staging_attempts/{ATTEMPT}"]["approval_artifact"] == approval_artifact

    progress = store.record_ingestion_progress(
        attempt_id=ATTEMPT, source=source, map_artifact=map_artifact,
        approval_artifact=approval_artifact, state="running", parent_count=1, child_count=0,
    )
    assert progress.state == "running"
    assert progress.completed_stage == "ingestion"
    assert progress.parent_count == 1 and progress.child_count == 0
    interrupted = store.record_ingestion_progress(
        attempt_id=ATTEMPT, source=source, map_artifact=map_artifact,
        approval_artifact=approval_artifact, state="failed", parent_count=1,
        child_count=1, correction_code="job_interrupted",
    )
    assert interrupted.state == "failed"
    assert interrupted.correction_codes == ["job_interrupted"]
    assert interrupted.error is not None and interrupted.error.code == "job_interrupted"

    retried, should_dispatch = store.claim_approval(
        attempt_id=ATTEMPT, expected_revision=interrupted.revision, approval=approval
    )
    assert retried.state == "queued" and should_dispatch
    staged = store.record_ingestion_progress(
        attempt_id=ATTEMPT, source=source, map_artifact=map_artifact,
        approval_artifact=approval_artifact, state="staged", parent_count=1,
        child_count=2, chunk_count=2, corpus_version_id="candidate-version",
    )
    assert staged.state == "staged"
    assert staged.staged_corpus_version_id == "candidate-version"
    assert staged.chunk_count == 2

    with pytest.raises(StagingAttemptConflict):
        store.record_ingestion_progress(
            attempt_id=ATTEMPT, source=source, map_artifact={**map_artifact, "generation": 99},
            approval_artifact=approval_artifact, state="running",
        )
