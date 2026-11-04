from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound

from app.api.corpus_staging import corpus_staging_upload_service
from app.contracts import CorpusStagingAttemptStatus, UserProfile
from app.corpus.staging import (
    FirestoreCorpusStagingAttemptStore,
    CorpusStagingUploadService,
    StagingAttemptConflict,
    StagingAttemptNotFound,
)
from app.identity import get_admin_user, get_current_profile
from app.ingestion.extract import ExtractionReport, PageProvenance
from app.main import app


ATTEMPT_ID = "attempt-324"
BUCKET = "private-corpus-bucket"
PDF = b"%PDF-1.7\nDMBOK Guide\n%%EOF"


def template(*entries: list[dict[str, Any]], document_title: str = "DMBOK Guide") -> bytes:
    return json.dumps(
        {
            "schema_version": "structure-template-v1",
            "document_title": document_title,
            "instructions": ["Map each passage to a section."],
            "entries": entries,
        }
    ).encode()


VALID_MAP = template(
    {"section_path": ["Chapter 1", "1.1 Purpose"], "page_start": 1, "page_end": 1},
    {"section_path": ["Chapter 1", "1.2 Roles"], "page_start": 2, "page_end": 2},
)


def extraction(
    *,
    page_sections: tuple[str | None, ...] = ("1.1 Purpose", "1.2 Roles"),
    failures: tuple[str, ...] = (),
) -> ExtractionReport:
    return ExtractionReport(
        source_uri="staged-source",
        source_sha256=sha256(PDF).hexdigest(),
        parser="test extractor",
        page_count=len(page_sections),
        pages=tuple(
            PageProvenance(
                page=index,
                section=section,
                text=("DMBOK Guide " if index == 1 else "") + (section or "unmapped passage"),
            )
            for index, section in enumerate(page_sections, start=1)
        ),
        failures=failures,
    )


class MemoryStore:
    def __init__(self, map_bytes: bytes = VALID_MAP) -> None:
        self.active_pointer = "active-corpus-v1"
        self.record: dict[str, Any] = {
            "attempt_id": ATTEMPT_ID,
            "actor_id": "admin-id",
            "revision": 3,
            "state": "validating",
            "source": {
                "bucket": BUCKET,
                "object_name": f"staging/{ATTEMPT_ID}/source.pdf",
                "generation": 17,
                "sha256": sha256(PDF).hexdigest(),
                "content_type": "application/pdf",
            },
            "structure": {
                "bucket": BUCKET,
                "object_name": f"staging/{ATTEMPT_ID}/structure.json",
                "generation": 23,
                "sha256": sha256(map_bytes).hexdigest(),
                "content_type": "application/json",
            },
        }

    def get(self, *, attempt_id: str) -> dict[str, Any] | None:
        return deepcopy(self.record) if attempt_id == ATTEMPT_ID else None

    def complete_validation(
        self,
        *,
        attempt_id: str,
        expected_revision: int,
        status,
        source_generation: int,
        structure_generation: int,
    ):
        if attempt_id != ATTEMPT_ID:
            raise StagingAttemptNotFound("missing")
        if self.record["state"] != "validating":
            return CorpusStagingAttemptStatus.model_validate(
                {key: value for key, value in self.record.items() if key in CorpusStagingAttemptStatus.model_fields}
            )
        if self.record["revision"] != expected_revision:
            raise StagingAttemptConflict("stale")
        result = status.model_copy(update={"revision": expected_revision + 1})
        self.record.update(result.model_dump(mode="json"))
        if result.preview is not None:
            self.record["preview_binding"] = {
                "preview_revision": result.revision,
                "source_generation": source_generation,
                "structure_generation": structure_generation,
            }
        return result


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str, generation: int | None) -> None:
        self.bucket = bucket
        self.name = name
        self.generation = generation

    def download_as_bytes(self, *, timeout: int) -> bytes:
        assert timeout == 60
        if self.bucket.generations[self.name] != self.generation:
            raise NotFound("object generation changed")
        return self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self, pdf: bytes, map_bytes: bytes) -> None:
        self.objects = {
            f"staging/{ATTEMPT_ID}/source.pdf": pdf,
            f"staging/{ATTEMPT_ID}/structure.json": map_bytes,
        }
        self.generations = {
            f"staging/{ATTEMPT_ID}/source.pdf": 17,
            f"staging/{ATTEMPT_ID}/structure.json": 23,
        }

    def blob(self, name: str, *, generation: int | None = None) -> FakeBlob:
        return FakeBlob(self, name, generation)


def profile(role: str) -> UserProfile:
    return UserProfile(
        user_id="admin-id" if role == "admin" else "user-id",
        email="test@example.com",
        username="Test",
        approval_state="approved",
        role=role,
        email_verified=True,
    )


def service_for(store: MemoryStore, *, pdf: bytes = PDF, map_bytes: bytes = VALID_MAP) -> CorpusStagingUploadService:
    return CorpusStagingUploadService(store, FakeBucket(pdf, map_bytes), BUCKET)


def test_valid_preview_persists_and_refresh_keeps_the_same_revision_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.corpus.staging as staging

    report = extraction()
    monkeypatch.setattr(staging, "extract_pdf", lambda *_args, **_kwargs: report)
    store = MemoryStore()
    service = service_for(store)

    reviewed = service.validate(attempt_id=ATTEMPT_ID)
    refreshed = service.status(attempt_id=ATTEMPT_ID)
    retried = service.validate(attempt_id=ATTEMPT_ID)

    assert reviewed.state == "awaiting_review"
    assert reviewed.revision == 4
    assert [(entry.section_path[-1], entry.page_start, entry.page_end, entry.match_status) for entry in reviewed.preview] == [
        ("1.1 Purpose", 1, 1, "matched"),
        ("1.2 Roles", 2, 2, "matched"),
    ]
    assert refreshed.model_dump(mode="json") == reviewed.model_dump(mode="json")
    assert retried.model_dump(mode="json") == reviewed.model_dump(mode="json")
    assert store.record["preview_binding"] == {
        "preview_revision": 4,
        "source_generation": 17,
        "structure_generation": 23,
    }
    assert store.active_pointer == "active-corpus-v1"


def test_wrong_document_pair_fails_without_persisting_a_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.corpus.staging as staging

    monkeypatch.setattr(staging, "extract_pdf", lambda *_args, **_kwargs: extraction())
    wrong_map = template(
        {"section_path": ["Chapter 1", "1.1 Purpose"], "page_start": 1, "page_end": 2},
        document_title="Different handbook",
    )
    store = MemoryStore(wrong_map)
    result = service_for(store, map_bytes=wrong_map).validate(attempt_id=ATTEMPT_ID)

    assert result.state == "failed"
    assert result.preview is None
    assert result.correction_codes == ["source_pair_mismatch"]
    assert store.active_pointer == "active-corpus-v1"


def test_missing_pinned_generation_fails_without_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.corpus.staging as staging

    monkeypatch.setattr(staging, "extract_pdf", lambda *_args, **_kwargs: extraction())
    store = MemoryStore()
    bucket = FakeBucket(PDF, VALID_MAP)
    bucket.generations[f"staging/{ATTEMPT_ID}/source.pdf"] = 18
    result = CorpusStagingUploadService(store, bucket, BUCKET).validate(attempt_id=ATTEMPT_ID)

    assert result.state == "failed"
    assert result.preview is None
    assert result.correction_codes == ["source_pair_mismatch"]
    assert store.active_pointer == "active-corpus-v1"


@pytest.mark.parametrize(
    ("map_bytes", "report", "expected_code"),
    [
        (
            template(
                {"section_path": ["Chapter 1", "1.1 Purpose"], "page_start": 1, "page_end": 1},
                {"section_path": ["Chapter 1", "1.2 Roles"], "page_start": 3, "page_end": 3},
            ),
            extraction(),
            "missing_page_coverage",
        ),
        (
            template(
                {"section_path": ["Chapter 1", "1.1 Purpose"], "page_start": 1, "page_end": 2},
                {"section_path": ["Appendix", "A.1 Notes"], "page_start": 1, "page_end": 2},
            ),
            extraction(),
            "ambiguous_heading",
        ),
        (
            VALID_MAP,
            extraction(page_sections=("Other section", "1.2 Roles")),
            "unmatched_passage",
        ),
    ],
)
def test_hierarchy_errors_fail_with_indexed_corrections_before_writes(
    monkeypatch: pytest.MonkeyPatch,
    map_bytes: bytes,
    report: ExtractionReport,
    expected_code: str,
) -> None:
    import app.corpus.staging as staging

    monkeypatch.setattr(staging, "extract_pdf", lambda *_args, **_kwargs: report)
    store = MemoryStore(map_bytes)
    result = service_for(store, map_bytes=map_bytes).validate(attempt_id=ATTEMPT_ID)

    assert result.state == "failed"
    assert expected_code in result.correction_codes
    assert result.preview is not None
    assert any(expected_code in entry.correction_codes for entry in result.preview)
    assert all(entry.entry_index == index for index, entry in enumerate(result.preview))
    assert store.active_pointer == "active-corpus-v1"


def test_validation_and_refresh_endpoints_are_admin_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.corpus.staging as staging

    monkeypatch.setattr(staging, "extract_pdf", lambda *_args, **_kwargs: extraction())
    service = service_for(MemoryStore())
    app.dependency_overrides.clear()
    app.dependency_overrides[corpus_staging_upload_service] = lambda: service
    client = TestClient(app)
    route = f"/api/admin/corpus-staging/attempts/{ATTEMPT_ID}"
    try:
        assert client.get(route).status_code == 401
        app.dependency_overrides[get_current_profile] = lambda: profile("user")
        assert client.get(route).status_code == 403
        app.dependency_overrides[get_admin_user] = lambda: profile("admin")
        assert client.post(f"{route}/validate").status_code == 200
        preview = client.get(route)
        assert preview.status_code == 200
        assert preview.json()["state"] == "awaiting_review"
        assert preview.json()["preview"][0]["entry_index"] == 0
    finally:
        app.dependency_overrides.clear()


def test_firestore_preview_write_is_revision_bound_and_readable() -> None:
    from test_evaluation_generations import FakeFirestore, run_fake_transaction

    database = FakeFirestore()
    store = FirestoreCorpusStagingAttemptStore(database, transaction_runner=run_fake_transaction)
    store.begin(
        attempt_id=ATTEMPT_ID,
        actor_id="admin-id",
        source_object_name=f"staging/{ATTEMPT_ID}/source.pdf",
        map_object_name=f"staging/{ATTEMPT_ID}/structure.json",
    )
    store.attach_artifact(
        attempt_id=ATTEMPT_ID,
        artifact_kind="source",
        artifact={"bucket": BUCKET, "object_name": f"staging/{ATTEMPT_ID}/source.pdf", "generation": 17, "sha256": sha256(PDF).hexdigest(), "content_type": "application/pdf"},
    )
    map_hash = sha256(VALID_MAP).hexdigest()
    store.attach_artifact(
        attempt_id=ATTEMPT_ID,
        artifact_kind="structure",
        artifact={"bucket": BUCKET, "object_name": f"staging/{ATTEMPT_ID}/structure.json", "generation": 23, "sha256": map_hash, "content_type": "application/json"},
    )
    result = CorpusStagingAttemptStatus(
        attempt_id=ATTEMPT_ID,
        revision=3,
        state="awaiting_review",
        completed_stage="review",
        preview=[
            {
                "entry_index": 0,
                "section_path": ["Chapter 1", "1.1 Purpose"],
                "page_start": 1,
                "page_end": 2,
                "match_status": "matched",
            }
        ],
    )

    saved = store.complete_validation(
        attempt_id=ATTEMPT_ID,
        expected_revision=3,
        status=result,
        source_generation=17,
        structure_generation=23,
    )
    record = store.get(attempt_id=ATTEMPT_ID)

    assert saved.revision == 4
    assert record["preview_binding"] == {
        "preview_revision": 4,
        "source_generation": 17,
        "structure_generation": 23,
    }
    assert CorpusStagingUploadService(store, FakeBucket(PDF, VALID_MAP), BUCKET).status(
        attempt_id=ATTEMPT_ID
    ).model_dump(mode="json") == saved.model_dump(mode="json")


