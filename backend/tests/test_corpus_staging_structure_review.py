from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from fastapi.testclient import TestClient
from google.api_core.exceptions import PreconditionFailed

from app.api.corpus_staging import corpus_staging_approval_service, corpus_staging_upload_service
from app.contracts import UserProfile
from app.contracts.v29 import ParsedBlock, ParsingArtifact
from app.corpus.staging import (
    CorpusStagingApprovalService,
    CorpusStagingUploadService,
    FirestoreCorpusStagingAttemptStore,
)
from app.identity import get_admin_user
from app.main import app
from test_evaluation_generations import FakeFirestore, run_fake_transaction


ATTEMPT = "block-review-test"
BUCKET = "private-corpus-bucket"
PDF = b"%PDF-1.7\nsource"
SOURCE_SHA = sha256(PDF).hexdigest()


class Blob:
    def __init__(self, bucket: "Bucket", name: str, generation: int | None = None) -> None:
        self.bucket, self.name = bucket, name
        self.generation = generation

    def upload_from_string(self, payload: bytes, *, content_type: str, if_generation_match: int, timeout: int) -> None:
        assert if_generation_match == 0 and timeout == 60
        if self.name in self.bucket.objects:
            raise PreconditionFailed("already exists")
        self.bucket.objects[self.name] = payload
        self.bucket.generations[self.name] = 100 + len(self.bucket.generations)
        self.generation = self.bucket.generations[self.name]

    def reload(self) -> None:
        self.generation = self.bucket.generations[self.name]

    def download_as_bytes(self, *, timeout: int) -> bytes:
        assert timeout == 60
        if self.bucket.generations[self.name] != self.generation:
            raise ValueError("wrong generation")
        return self.bucket.objects[self.name]


class Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.generations: dict[str, int] = {}

    def blob(self, name: str, generation: int | None = None) -> Blob:
        return Blob(self, name, generation)


class Dispatcher:
    def __init__(self) -> None:
        self.environments: list[dict[str, str]] = []

    def dispatch(self, **kwargs) -> None:
        self.environments.append(kwargs)


def profile() -> UserProfile:
    return UserProfile(
        user_id="admin-1", email="admin@example.com", username="Admin",
        approval_state="approved", role="admin", email_verified=True,
    )


def setup(*, unclassified: bool = False):
    blocks = [
        ParsedBlock(block_id="b1", physical_page=21, reading_order=0, kind="text", raw_text="Opening context")
    ] if unclassified else []
    blocks.extend([
        ParsedBlock(block_id="b2", physical_page=21, reading_order=len(blocks), kind="text", raw_text="# Chapter 1"),
        ParsedBlock(block_id="b3", physical_page=22, reading_order=len(blocks) + 1, kind="text", raw_text="body " + "x" * 700),
        ParsedBlock(block_id="b4", physical_page=23, reading_order=len(blocks) + 2, kind="text", raw_text="## Section 1.1"),
    ])
    parsing = ParsingArtifact(
        source_generation="19", source_sha256=SOURCE_SHA, output_sha256="c" * 64,
        blocks=tuple(blocks),
    )
    raw = parsing.model_dump_json().encode()
    parse_path = f"staging/{ATTEMPT}/parsing/{parsing.output_sha256}.json"
    bucket = Bucket()
    bucket.objects[parse_path] = raw
    bucket.generations[parse_path] = 71
    database = FakeFirestore()
    store = FirestoreCorpusStagingAttemptStore(database, transaction_runner=run_fake_transaction)
    store.begin(
        attempt_id=ATTEMPT, actor_id="admin-1", source_object_name=f"staging/{ATTEMPT}/source.pdf",
        map_object_name=None,
    )
    source = {
        "bucket": BUCKET, "object_name": f"staging/{ATTEMPT}/source.pdf", "generation": 19,
        "sha256": SOURCE_SHA, "content_type": "application/pdf",
    }
    store.attach_artifact(attempt_id=ATTEMPT, artifact_kind="source", artifact=source)
    parsing_identity = {
        "object_name": parse_path,
        "source_generation": "19",
        "source_sha256": SOURCE_SHA,
        "sha256": parsing.output_sha256,
        "artifact_sha256": sha256(raw).hexdigest(),
        "artifact_generation": 71,
        "remote_job_id": "llama-job-1",
    }
    database.data[f"corpus_staging_attempts/{ATTEMPT}"].update(
        parsing_state="awaiting_review", parsing_artifact=parsing_identity,
    )
    upload = CorpusStagingUploadService(store, bucket, BUCKET)
    dispatcher = Dispatcher()
    approval = CorpusStagingApprovalService(store, bucket, BUCKET, dispatcher)
    app.dependency_overrides.clear()
    app.dependency_overrides[get_admin_user] = profile
    app.dependency_overrides[corpus_staging_upload_service] = lambda: upload
    app.dependency_overrides[corpus_staging_approval_service] = lambda: approval
    return TestClient(app), store, database, bucket, parsing, dispatcher


def test_proposal_uses_saved_blocks_returns_bounded_excerpts_and_approval_is_revision_bound() -> None:
    client, store, database, bucket, parsing, dispatcher = setup()
    route = f"/api/admin/corpus-staging/attempts/{ATTEMPT}"
    try:
        proposed = client.post(f"{route}/structure/propose")
        assert proposed.status_code == 200
        body = proposed.json()
        assert body["state"] == "awaiting_review"
        assert body["parsing_output_sha256"] == parsing.output_sha256
        assert [entry["block_ids"] for entry in body["preview"]] == [["b2", "b3"], ["b4"]]
        assert all(len(entry["excerpt"]) <= 500 for entry in body["preview"])
        record = database.data[f"corpus_staging_attempts/{ATTEMPT}"]
        first_map = deepcopy(record["structure"])
        first_bytes = bucket.blob(first_map["object_name"], first_map["generation"]).download_as_bytes(timeout=60)
        assert json.loads(first_bytes)["parsing_output_sha256"] == parsing.output_sha256

        stale = client.put(f"{route}/structure", json={
            "expected_revision": body["revision"],
            "entries": [
                {"section_path": ["Chapter 1"], "block_ids": ["b2", "b3"]},
                {"section_path": ["Chapter 1", "1.1 Section"], "block_ids": ["b4"]},
            ],
        })
        assert stale.status_code == 200
        corrected = stale.json()
        second_map = deepcopy(database.data[f"corpus_staging_attempts/{ATTEMPT}"]["structure"])
        assert second_map["object_name"] != first_map["object_name"]
        second_bytes = bucket.blob(second_map["object_name"], second_map["generation"]).download_as_bytes(timeout=60)
        assert first_bytes != second_bytes

        out_of_date = client.post(f"{route}/approve?expected_revision={body['revision']}")
        assert out_of_date.status_code == 409
        queued = client.post(f"{route}/approve?expected_revision={corrected['revision']}")
        repeated = client.post(f"{route}/approve?expected_revision={corrected['revision']}")
        assert queued.status_code == repeated.status_code == 200
        assert queued.json()["state"] == repeated.json()["state"] == "queued"
        assert len(dispatcher.environments) == 1
        dispatched = dispatcher.environments[0]
        assert dispatched["source"]["generation"] == 19
        assert dispatched["map_artifact"] == second_map
        assert dispatched["parsing_artifact"]["output_sha256"] == parsing.output_sha256
    finally:
        app.dependency_overrides.clear()


def test_incomplete_duplicate_or_unclassified_block_assignments_cannot_dispatch() -> None:
    client, _, _, _, _, dispatcher = setup(unclassified=True)
    route = f"/api/admin/corpus-staging/attempts/{ATTEMPT}"
    try:
        proposed = client.post(f"{route}/structure/propose").json()
        assert "unmatched_passage" in proposed["correction_codes"]
        assert client.post(f"{route}/approve?expected_revision={proposed['revision']}").status_code == 409
        for entries in (
            [{"section_path": ["Chapter 1"], "block_ids": ["b2", "b3", "b4"]}],
            [
                {"section_path": ["Chapter 1"], "block_ids": ["b1", "b2", "b3", "b4"]},
                {"section_path": ["Chapter 2"], "block_ids": ["b4"]},
            ],
        ):
            invalid = client.put(f"{route}/structure", json={
                "expected_revision": proposed["revision"], "entries": entries,
            })
            assert invalid.status_code == 409
        assert dispatcher.environments == []
    finally:
        app.dependency_overrides.clear()
