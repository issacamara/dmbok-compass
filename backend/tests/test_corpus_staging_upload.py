"""Paired corpus staging upload keeps artifact identity server-owned."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.contracts import ApiError, CorpusStagingAttemptStatus, UserProfile
from app.api.corpus_staging import (
    _configured_hosted_parsing_dispatch,
    corpus_staging_upload_service,
    hosted_parsing_dispatch,
)
from app.corpus.staging import (
    CorpusStagingUploadService,
    FirestoreCorpusStagingAttemptStore,
    MAX_STAGING_MAP_BYTES,
    MAX_STAGING_PDF_BYTES,
    StagingUploadValidationError,
    validate_staging_pdf,
    validate_staging_files,
)
from app.identity import get_admin_user, get_current_profile
from app.ingestion.hosted_parsing import ParsingAttemptNotFound, ParsingOutcomeUnknown
from app.main import app
from test_evaluation_generations import FakeFirestore


PDF = b"%PDF-1.7\nprivate DMBOK contents\n%%EOF"
STRUCTURE = json.dumps(
    {
        "schema_version": "structure-template-v1",
        "document_title": "DMBOK",
        "instructions": ["Map each passage to a section."],
        "entries": [{"section_path": ["Data Governance"], "page_start": 1, "page_end": 1}],
    }
).encode()


class MemoryStore:
    def __init__(self) -> None:
        self.attempts: dict[str, dict[str, object]] = {}
        self.active_pointer = "active-corpus-v1"

    def begin(self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str | None) -> int:
        self.attempts[attempt_id] = {
            "attempt_id": attempt_id,
            "actor_id": actor_id,
            "state": "validating",
            "revision": 1,
            "source_object_name": source_object_name,
            "map_object_name": map_object_name,
            "source": None,
            "structure": None,
        }
        return 1

    def get(self, *, attempt_id: str):
        return deepcopy(self.attempts[attempt_id]) if attempt_id in self.attempts else None

    def attach_artifact(self, *, attempt_id: str, artifact_kind: str, artifact: dict[str, object]) -> int:
        attempt = self.attempts[attempt_id]
        if attempt.get(artifact_kind) == artifact:
            return int(attempt["revision"])
        revision = int(attempt["revision"]) + 1
        attempt[artifact_kind] = deepcopy(artifact)
        attempt["revision"] = revision
        return revision

    def fail(self, *, attempt_id: str, error: ApiError):
        from app.contracts import CorpusStagingAttemptStatus

        attempt = self.attempts[attempt_id]
        revision = int(attempt["revision"]) + 1
        attempt.update({"state": "failed", "error": error.model_dump(mode="json"), "revision": revision})
        return CorpusStagingAttemptStatus(attempt_id=attempt_id, revision=revision, state="failed", error=error)


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str) -> None:
        self.bucket = bucket
        self.name = name
        self.generation: int | None = None

    def upload_from_string(
        self,
        content: bytes,
        *,
        content_type: str,
        if_generation_match: int,
        timeout: int,
    ) -> None:
        assert timeout == 60
        if if_generation_match != 0 or self.name in self.bucket.objects:
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("object already exists")
        if self.bucket.fail_name and self.name.endswith(self.bucket.fail_name):
            raise RuntimeError(f"failed to write {self.name}")
        self.bucket.objects[self.name] = {"content": content, "content_type": content_type}
        self.generation = 71

    def reload(self, *, timeout: int) -> None:
        assert timeout == 30
        self.generation = 71

    def download_as_bytes(self, *, timeout: int) -> bytes:
        assert timeout == 60
        return self.bucket.objects[self.name]["content"]


class FakeBucket:
    def __init__(self, *, fail_name: str | None = None) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.fail_name = fail_name

    def blob(self, name: str, generation: int | None = None) -> FakeBlob:
        return FakeBlob(self, name)


def profile(role: str) -> UserProfile:
    return UserProfile(
        user_id="admin-id" if role == "admin" else "user-id",
        email="test@example.com",
        username="Test",
        approval_state="approved",
        role=role,
        email_verified=True,
    )


def setup_client(service: CorpusStagingUploadService) -> TestClient:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_admin_user] = lambda: profile("admin")
    app.dependency_overrides[corpus_staging_upload_service] = lambda: service
    return TestClient(app)


def files(
    pdf: bytes = PDF,
    structure: bytes = STRUCTURE,
    *,
    pdf_name: str = "dmbok.pdf",
    pdf_type: str = "application/pdf",
    structure_name: str = "structure.json",
    structure_type: str = "application/json",
):
    return {
        "source_pdf": (pdf_name, pdf, pdf_type),
        "structure_json": (structure_name, structure, structure_type),
    }


def test_paired_upload_persists_one_attempt_with_immutable_generations_and_hashes() -> None:
    database = FakeFirestore()
    store = FirestoreCorpusStagingAttemptStore(database)
    bucket = FakeBucket()
    service = CorpusStagingUploadService(store, bucket, "private-corpus-bucket")
    client = setup_client(service)
    try:
        response = client.post("/api/admin/corpus-staging/attempts", files=files())
        assert response.status_code == 201
        body = response.json()
        assert body["state"] == "validating"
        assert body["revision"] == 3
        assert set(body) == {
            "attempt_id", "revision", "state", "completed_stage", "preview",
            "correction_codes", "parent_count", "child_count", "chunk_count",
            "staged_corpus_version_id", "hosted_parsing_state", "parsing_output_sha256", "error",
        }
        assert body["parsing_output_sha256"] is None
        assert not any("uri" in key for key in body)

        records = {
            path: value
            for path, value in database.data.items()
            if path.startswith("corpus_staging_attempts/")
        }
        assert len(records) == 1
        record = next(iter(records.values()))
        assert record["actor_id"] == "admin-id"
        assert record["state"] == "validating"
        assert record["revision"] == 3
        assert record["source"]["generation"] == 71
        assert record["source"]["sha256"] == sha256(PDF).hexdigest()
        assert record["structure"]["generation"] == 71
        assert record["structure"]["sha256"] == sha256(STRUCTURE).hexdigest()
        assert all(name.startswith("staging/" + body["attempt_id"] + "/") for name in bucket.objects)
    finally:
        app.dependency_overrides.clear()


def test_pdf_only_upload_is_idempotent_and_keeps_paired_path_compatible() -> None:
    database = FakeFirestore()
    store = FirestoreCorpusStagingAttemptStore(database)
    bucket = FakeBucket()
    service = CorpusStagingUploadService(store, bucket, "private-corpus-bucket")
    client = setup_client(service)
    try:
        first = client.post(
            "/api/admin/corpus-staging/attempts/pdf",
            files={"source_pdf": ("dmbok.pdf", PDF, "application/pdf")},
        )
        second = client.post(
            "/api/admin/corpus-staging/attempts/pdf",
            files={"source_pdf": ("dmbok.pdf", PDF, "application/pdf")},
        )
        assert first.status_code == second.status_code == 201
        assert first.json()["attempt_id"] == second.json()["attempt_id"]
        records = [value for path, value in database.data.items() if path.startswith("corpus_staging_attempts/")]
        assert len(records) == 1
        assert records[0]["source"]["sha256"] == sha256(PDF).hexdigest()
        assert records[0]["structure"] is None
        assert len(bucket.objects) == 1

        paired = client.post("/api/admin/corpus-staging/attempts", files=files())
        assert paired.status_code == 201
        assert paired.json()["attempt_id"] != first.json()["attempt_id"]
        assert len([path for path in database.data if path.startswith("corpus_staging_attempts/")]) == 2
    finally:
        app.dependency_overrides.clear()


def test_concurrent_source_create_and_object_collision_do_not_fail_attempt() -> None:
    class ConcurrentStore(MemoryStore):
        def begin(self, **kwargs):
            revision = super().begin(**kwargs)
            if not getattr(self, "raced", False):
                self.raced = True
                raise RuntimeError("document was created by a concurrent request")
            return revision

    store = ConcurrentStore()
    bucket = FakeBucket()
    actor_id = "admin-id"
    attempt_id = sha256(f"{actor_id}\0{sha256(PDF).hexdigest()}".encode()).hexdigest()[:32]
    object_name = f"staging/{attempt_id}/source.pdf"
    bucket.objects[object_name] = {"content": PDF, "content_type": "application/pdf"}

    result = CorpusStagingUploadService(store, bucket, "private-corpus-bucket").submit_pdf(
        actor_id=actor_id, pdf_bytes=PDF
    )

    assert result.attempt_id == attempt_id
    assert result.state == "validating"
    assert store.attempts[attempt_id]["source"]["sha256"] == sha256(PDF).hexdigest()
    assert store.attempts[attempt_id]["state"] == "validating"


def test_parse_route_dispatches_queued_work_and_rejects_unresolved_retry():
    class Checkpoints:
        calls = 0

        def enqueue(self, attempt_id):
            self.calls += 1
            if self.calls == 1:
                return {"parsing_state": "queued"}
            raise ParsingOutcomeUnknown("submission requires reconciliation")

    class Dispatcher:
        calls = 0

        def dispatch(self):
            self.calls += 1

    class StatusService:
        def status(self, *, attempt_id):
            return CorpusStagingAttemptStatus(
                attempt_id=attempt_id, revision=1, state="validating", hosted_parsing_state="queued"
            )

    checkpoints, dispatcher = Checkpoints(), Dispatcher()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_admin_user] = lambda: profile("admin")
    app.dependency_overrides[hosted_parsing_dispatch] = lambda: (checkpoints, dispatcher)
    app.dependency_overrides[corpus_staging_upload_service] = StatusService
    try:
        client = TestClient(app)
        first = client.post("/api/admin/corpus-staging/attempts/attempt-1/parse")
        retry = client.post("/api/admin/corpus-staging/attempts/attempt-1/parse")
        assert first.status_code == 200
        assert first.json()["hosted_parsing_state"] == "queued"
        assert retry.status_code == 409
        assert retry.json()["detail"]["code"] == "parsing_outcome_unresolved"
        assert dispatcher.calls == 1
    finally:
        app.dependency_overrides.clear()


def test_parse_route_returns_not_found_for_missing_attempt():
    class Checkpoints:
        def enqueue(self, attempt_id):
            raise ParsingAttemptNotFound("staging attempt is unavailable")

    class Dispatcher:
        def dispatch(self):
            raise AssertionError("missing attempts must not be dispatched")

    app.dependency_overrides.clear()
    app.dependency_overrides[get_admin_user] = lambda: profile("admin")
    app.dependency_overrides[hosted_parsing_dispatch] = lambda: (Checkpoints(), Dispatcher())
    app.dependency_overrides[corpus_staging_upload_service] = lambda: object()
    try:
        client = TestClient(app)
        response = client.post("/api/admin/corpus-staging/attempts/missing/parse")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "staging_attempt_not_found"
    finally:
        app.dependency_overrides.clear()


def test_parse_route_returns_503_when_parser_job_configuration_is_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("PARSER_JOB_NAME", raising=False)
    _configured_hosted_parsing_dispatch.cache_clear()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_admin_user] = lambda: profile("admin")
    app.dependency_overrides[corpus_staging_upload_service] = lambda: object()
    try:
        client = TestClient(app)
        response = client.post("/api/admin/corpus-staging/attempts/attempt-1/parse")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "parsing_dispatch_unavailable"
    finally:
        app.dependency_overrides.clear()
        _configured_hosted_parsing_dispatch.cache_clear()


@pytest.mark.parametrize(
    ("name", "content_type", "content", "expected"),
    [("book.txt", "application/pdf", PDF, 415), ("book.pdf", "application/pdf", b"bad", 422)],
)
def test_pdf_only_upload_rejects_invalid_file_metadata_or_signature(name, content_type, content, expected):
    client = setup_client(CorpusStagingUploadService(MemoryStore(), FakeBucket(), "private-corpus-bucket"))
    try:
        response = client.post(
            "/api/admin/corpus-staging/attempts/pdf",
            files={"source_pdf": (name, content, content_type)},
        )
        assert response.status_code == expected
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (files(pdf=b"not pdf"), 422),
        (files(pdf_type="application/octet-stream"), 415),
        (files(structure=b"not json"), 422),
        (files(structure_type="text/plain"), 415),
    ],
)
def test_invalid_pair_is_rejected_before_creating_attempt_or_objects(payload, expected_status: int) -> None:
    store = MemoryStore()
    bucket = FakeBucket()
    client = setup_client(CorpusStagingUploadService(store, bucket, "private-corpus-bucket"))
    try:
        response = client.post("/api/admin/corpus-staging/attempts", files=payload)
        assert response.status_code == expected_status
        assert store.attempts == {}
        assert bucket.objects == {}
        assert store.active_pointer == "active-corpus-v1"
    finally:
        app.dependency_overrides.clear()


def test_missing_and_oversized_files_do_not_create_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.corpus_staging as api

    store = MemoryStore()
    bucket = FakeBucket()
    client = setup_client(CorpusStagingUploadService(store, bucket, "private-corpus-bucket"))
    monkeypatch.setattr(api, "MAX_STAGING_PDF_BYTES", 8)
    try:
        missing = client.post(
            "/api/admin/corpus-staging/attempts",
            files={"source_pdf": ("dmbok.pdf", PDF, "application/pdf")},
        )
        oversized = client.post("/api/admin/corpus-staging/attempts", files=files())
        assert missing.status_code == 422
        assert oversized.status_code == 413
        assert store.attempts == {}
        assert bucket.objects == {}
        assert store.active_pointer == "active-corpus-v1"
    finally:
        app.dependency_overrides.clear()


def test_unauthorized_callers_cannot_create_an_attempt() -> None:
    database = FakeFirestore()
    bucket = FakeBucket()
    service = CorpusStagingUploadService(
        FirestoreCorpusStagingAttemptStore(database), bucket, "private-corpus-bucket"
    )
    app.dependency_overrides.clear()
    app.dependency_overrides[corpus_staging_upload_service] = lambda: service
    try:
        client = TestClient(app)
        anonymous = client.post("/api/admin/corpus-staging/attempts", files=files())
        app.dependency_overrides[get_current_profile] = lambda: profile("user")
        unauthorized = client.post("/api/admin/corpus-staging/attempts", files=files())
        assert anonymous.status_code == 401
        assert unauthorized.status_code == 403
        assert database.data == {}
        assert bucket.objects == {}
    finally:
        app.dependency_overrides.clear()


def test_partial_storage_failure_is_durable_safe_and_does_not_change_active_corpus(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MemoryStore()
    bucket = FakeBucket(fail_name="structure.json")
    service = CorpusStagingUploadService(store, bucket, "private-corpus-bucket")
    client = setup_client(service)
    caplog.set_level(logging.DEBUG)
    try:
        response = client.post("/api/admin/corpus-staging/attempts", files=files())
        assert response.status_code == 201
        assert response.json()["state"] == "failed"
        assert response.json()["error"]["code"] == "staging_storage_failed"
        assert len(store.attempts) == 1
        failed = next(iter(store.attempts.values()))
        assert failed["state"] == "failed"
        assert failed["source"]["sha256"] == sha256(PDF).hexdigest()
        assert failed["structure"] is None
        assert store.active_pointer == "active-corpus-v1"
        assert failed["source_object_name"] not in caplog.text
        assert failed["map_object_name"] not in caplog.text
        assert "private-corpus-bucket" not in caplog.text
        assert sha256(PDF).hexdigest() not in caplog.text
        assert "private DMBOK contents" not in caplog.text
    finally:
        app.dependency_overrides.clear()


def test_structure_template_contract_and_upload_limits_are_checked() -> None:
    with pytest.raises(StagingUploadValidationError, match="allowed size"):
        validate_staging_files(
            pdf_filename="dmbok.pdf",
            pdf_content_type="application/pdf",
            pdf_bytes=b"%PDF-" + b"x" * MAX_STAGING_PDF_BYTES,
            map_filename="structure.json",
            map_content_type="application/json",
            map_bytes=STRUCTURE,
        )
    with pytest.raises(StagingUploadValidationError, match="allowed size"):
        validate_staging_files(
            pdf_filename="dmbok.pdf",
            pdf_content_type="application/pdf",
            pdf_bytes=PDF,
            map_filename="structure.json",
            map_content_type="application/json",
            map_bytes=b" " * MAX_STAGING_MAP_BYTES + b"{}",
        )
