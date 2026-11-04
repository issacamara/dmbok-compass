"""Paired corpus staging upload keeps artifact identity server-owned."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.contracts import ApiError, UserProfile
from app.api.corpus_staging import corpus_staging_upload_service
from app.corpus.staging import (
    CorpusStagingUploadService,
    FirestoreCorpusStagingAttemptStore,
    MAX_STAGING_MAP_BYTES,
    MAX_STAGING_PDF_BYTES,
    StagingUploadValidationError,
    validate_staging_files,
)
from app.identity import get_admin_user, get_current_profile
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

    def begin(self, *, attempt_id: str, actor_id: str, source_object_name: str, map_object_name: str) -> int:
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

    def attach_artifact(self, *, attempt_id: str, artifact_kind: str, artifact: dict[str, object]) -> int:
        attempt = self.attempts[attempt_id]
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
            raise RuntimeError("object is not create-only")
        if self.bucket.fail_name and self.name.endswith(self.bucket.fail_name):
            raise RuntimeError(f"failed to write {self.name}")
        self.bucket.objects[self.name] = {"content": content, "content_type": content_type}
        self.generation = 71


class FakeBucket:
    def __init__(self, *, fail_name: str | None = None) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.fail_name = fail_name

    def blob(self, name: str) -> FakeBlob:
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
            "staged_corpus_version_id", "error",
        }
        assert not any("sha" in key or "uri" in key for key in body)

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
