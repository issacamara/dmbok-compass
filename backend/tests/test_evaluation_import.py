from __future__ import annotations

import json
import logging
from hashlib import sha256

from fastapi import status
from fastapi.testclient import TestClient

from app.api.evaluation_datasets import evaluation_generation_store
from app.api.evaluations import evaluation_service
from app.corpus.selection import uploaded_corpus_version_id
from app.contracts import DatasetHead, EvaluationRowV2, EvaluationRun, UserProfile
from app.evaluation import (
    EvaluationCorpusReadinessError,
    EvaluationDispatchError,
    FirestoreEvaluationGenerationStore,
)
from app.evaluation.legacy_import_binding import bind_legacy_imports
from app.identity import IdentityError, get_admin_user
from app.main import app
from test_evaluation_generations import FakeFirestore, run_fake_transaction
from test_candidate_generation import source_records


def _admin() -> UserProfile:
    return UserProfile(
        user_id="admin", email="admin@example.com", username="admin",
        approval_state="approved", role="admin", email_verified=True,
    )


def _store(client: FakeFirestore) -> FirestoreEvaluationGenerationStore:
    return FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)


def _record(**overrides: object) -> dict[str, object]:
    return {
        "question": "What is metadata?",
        "category": "definitions",
        "expected_answer_or_rubric": "A definition of metadata.",
        "supporting_passage": "Metadata describes data.",
        **overrides,
    }


class _FakeEvaluationService:
    def __init__(self) -> None:
        self.calls = 0

    def launch(self, *, item_ids: list[str] | None) -> EvaluationRun:
        assert item_ids is None
        self.calls += 1
        return EvaluationRun(
            run_id=f"run-{self.calls}",
            dataset_version_id="server-generated",
            corpus_version_id="uploaded-corpus",
            status="queued",
            configuration_version_id="runtime-config",
            model_version_id="runtime-model",
        )


class _DispatchFailureService(_FakeEvaluationService):
    def launch(self, *, item_ids: list[str] | None) -> EvaluationRun:
        raise EvaluationDispatchError(403, "PERMISSION_DENIED", "Cloud Run rejected the caller permissions.")


class _CorpusNotReadyService(_FakeEvaluationService):
    def launch(self, *, item_ids: list[str] | None) -> EvaluationRun:
        raise EvaluationCorpusReadinessError("evaluation corpus provenance evidence is unavailable")


def _service() -> _FakeEvaluationService:
    service = _FakeEvaluationService()
    app.dependency_overrides[evaluation_service] = lambda: service
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "uploaded-corpus"
    return service


def _post(client: TestClient, records: object):
    if evaluation_service not in app.dependency_overrides:
        _service()
    else:
        app.dependency_overrides.setdefault(uploaded_corpus_version_id, lambda: "uploaded-corpus")
    return client.post(
        "/api/admin/evaluation-datasets/import",
        content=json.dumps(records),
        headers={"Content-Type": "application/json"},
    )


def test_imports_valid_records_and_reports_invalid_indices_without_content() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    service = _service()
    try:
        response = _post(TestClient(app), [_record(), _record(category="other"), _record(question="  ")])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert body["status"] == "accepted"
    assert (body["submitted_count"], body["imported_count"], body["skipped_count"]) == (3, 1, 2)
    assert [(error["index"], error["code"]) for error in body["validation_errors"]] == [
        (1, "invalid_category"), (2, "missing_question")
    ]
    assert "What is metadata?" not in json.dumps(body)
    active_id = firestore.data["evaluation_dataset_state/active"]["generation_id"]
    annotations = [data for path, data in firestore.data.items() if "/annotations/" in path]
    assert active_id == body["generation_id"]
    assert annotations[0]["supporting_passage"] == "Metadata describes data."
    assert body["run_id"] == "run-1"
    assert service.calls == 1


def test_legacy_import_binds_unique_parse_span_and_keeps_unmatched_indexed() -> None:
    children, parents = source_records()
    records = (
        (0, "import-0", "What is data quality?", "definitions", "Explain it.", "Data quality requires clear ownership."),
        (1, "import-1", "What is governance?", "definitions", "Explain it.", "not present in this corpus"),
    )
    bound, errors = bind_legacy_imports(records, parents=parents, children=children)

    assert [item.index for item in bound] == [0]
    assert bound[0].row.source_spans[0].span_id == "span-21-0"
    assert bound[0].row.source_spans[0].block_id == "block-21-0"
    assert bound[0].row.source_spans[0].raw_text == "Data quality requires clear ownership."
    assert bound[0].row.relevant_child_ids == (children[0].chunk_id,)
    assert bound[0].row.parent_ids == (parents[0].parent_id,)
    assert [(error.index, error.code) for error in errors] == [(1, "unmatched_supporting_passage")]


def test_legacy_import_rejects_ambiguous_or_cross_corpus_matches() -> None:
    children, parents = source_records()
    duplicate = children[0].model_copy(update={"chunk_id": "another-child", "child_ordinal": 1})
    other_corpus = children[0].model_copy(update={"chunk_id": "foreign-child", "corpus_version_id": "other-corpus"})
    record = (0, "import-0", "What is data quality?", "definitions", "Explain it.", "Data quality requires clear ownership.")

    ambiguous, errors = bind_legacy_imports((record,), parents=parents, children=(children[0], duplicate, other_corpus))

    assert not ambiguous
    assert [(error.index, error.code) for error in errors] == [(0, "ambiguous_supporting_passage")]


def test_import_publishes_only_uniquely_bound_rows_to_row_v2() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    children, parents = source_records()
    repository.get_parse_derived_records = lambda *, corpus_version_id: (parents, children)
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    service = _service()
    try:
        response = _post(TestClient(app), [_record(supporting_passage="Data quality requires clear ownership."), _record(
            question="No corpus match?", supporting_passage="not in the source"
        )])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert body["row_v2_head"]["corpus_version_id"] == "uploaded-corpus"
    assert body["row_v2_head"]["revision"] == 1
    assert [(error["index"], error["code"]) for error in body["binding_errors"]] == [
        (1, "unmatched_supporting_passage")
    ]
    revision = repository.get_revision(repository.corpus_head("uploaded-corpus"))
    assert len(revision) == 1
    assert revision[0].origin == "imported"
    assert revision[0].source_spans[0].block_id == "block-21-0"
    assert revision[0].relevant_child_ids == (children[0].chunk_id,)
    assert revision[0].parent_ids == (parents[0].parent_id,)
    assert service.calls == 1


def test_admin_row_add_remove_replay_conflict_and_empty_set_protection() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    initial = EvaluationRowV2(
        item_id="initial-row", origin="admin", question="What is metadata?", category="definitions",
        expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse outside scope.",
        expected_refusal=True, scope_rationale="Outside this small fixture corpus.",
    )
    first = repository.stage_revision("uploaded-corpus", 1, (initial,))
    repository.activate_revision(first, expected_revision=None)
    children, parents = source_records()
    repository.get_parse_derived_records = lambda *, corpus_version_id: (parents, children)
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "uploaded-corpus"
    client = TestClient(app)
    add_payload = {
        "expected_revision": 1,
        "operation_key": "admin-add-one",
        "question": "How does stewardship support governance?",
        "category": "explanations",
        "expected_outcome": "qualified",
        "expected_answer_or_rubric": "Explain how the concepts relate.",
        "supporting_passage": "Data quality requires clear ownership.",
    }
    try:
        added = client.post("/api/admin/evaluation-datasets/rows", json=add_payload)
        replay = client.post("/api/admin/evaluation-datasets/rows", json=add_payload)
        reused_key = client.post("/api/admin/evaluation-datasets/rows", json={
            **add_payload, "question": "A different manual question?"
        })
        added_head = DatasetHead.model_validate(added.json()["head"])
        added_rows = repository.get_revision(added_head)
        removed = client.post(
            "/api/admin/evaluation-datasets/rows/initial-row/remove",
            json={"expected_revision": 2, "operation_key": "admin-remove-one"},
        )
        empty = client.post(
            f"/api/admin/evaluation-datasets/rows/{added_rows[1].item_id}/remove",
            json={"expected_revision": 3, "operation_key": "admin-remove-last"},
        )
        stale = client.post("/api/admin/evaluation-datasets/rows", json={
            **add_payload, "expected_revision": 2, "operation_key": "admin-stale-add"
        })
    finally:
        app.dependency_overrides.clear()

    assert added.status_code == status.HTTP_201_CREATED
    assert replay.status_code == status.HTTP_200_OK
    assert replay.json()["head"] == added.json()["head"]
    assert reused_key.status_code == status.HTTP_409_CONFLICT
    assert removed.status_code == status.HTTP_200_OK
    assert empty.status_code == status.HTTP_409_CONFLICT
    assert stale.status_code == status.HTTP_409_CONFLICT
    removed_head = DatasetHead.model_validate(removed.json()["head"])
    assert [row.item_id for row in repository.get_revision(first)] == ["initial-row"]
    assert [row.origin for row in repository.get_revision(added_head)] == ["admin", "admin"]
    assert len(repository.get_revision(removed_head)) == 1
    assert repository.corpus_head("uploaded-corpus") == removed_head
    exclusion, question = repository.get_dataset_exclusion_questions()[0]
    assert question == "What is metadata?"
    assert exclusion.actor_id == "admin"
    assert exclusion.revision == removed_head.revision


def test_rejections_and_storage_failures_leave_existing_active_generation_unchanged() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    first = _record()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    _service()
    try:
        created = _post(TestClient(app), [first])
        active_id = created.json()["generation_id"]
        rejected = _post(TestClient(app), [_record(question="")])
        children, parents = source_records()
        repository.get_parse_derived_records = lambda *, corpus_version_id: (parents, children)
        firestore.fail_next_batch = True
        failed = _post(TestClient(app), [_record(
            question="A different question",
            supporting_passage="Data quality requires clear ownership.",
        )])
    finally:
        app.dependency_overrides.clear()

    assert rejected.status_code == status.HTTP_200_OK
    assert rejected.json()["status"] == "rejected"
    assert failed.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert firestore.data["evaluation_dataset_state/active"]["generation_id"] == active_id
    assert repository.corpus_head("uploaded-corpus") is None


def test_dispatch_failure_exposes_only_bounded_provider_diagnostics() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    app.dependency_overrides[evaluation_service] = lambda: _DispatchFailureService()
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "uploaded-corpus"
    try:
        response = _post(TestClient(app), [_record(question="private question")])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "code": "evaluation_dispatch_failed",
        "message": "The evaluation run could not be started.",
        "provider_status": 403,
        "provider_code": "PERMISSION_DENIED",
        "provider_message": "Cloud Run rejected the caller permissions.",
    }
    assert "private question" not in response.text


def test_corpus_readiness_failure_is_distinct_from_dispatch_failure() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    app.dependency_overrides[evaluation_service] = lambda: _CorpusNotReadyService()
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "uploaded-corpus"
    try:
        response = _post(TestClient(app), [_record()])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"] == {
        "code": "evaluation_corpus_not_ready",
        "message": "The evaluation corpus is not ready for provenance-bound evaluation.",
    }


def test_import_enforces_bounds_and_administrator_access() -> None:
    app.dependency_overrides[get_admin_user] = lambda: (_ for _ in ()).throw(
        IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)
    )
    try:
        denied = _post(TestClient(app), [_record()])
    finally:
        app.dependency_overrides.clear()
    assert denied.status_code == status.HTTP_403_FORBIDDEN

    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(FakeFirestore())
    _service()
    try:
        malformed = TestClient(app).post("/api/admin/evaluation-datasets/import", content=b"{")
        over_limit = _post(TestClient(app), [_record()] * 501)
    finally:
        app.dependency_overrides.clear()
    assert malformed.status_code == status.HTTP_400_BAD_REQUEST
    assert over_limit.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_upload_and_run_rejects_technical_fields_without_mutation() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    service = _service()
    try:
        response = _post(TestClient(app), [_record(corpus_version_id="client-selected")])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "rejected"
    assert response.json()["validation_errors"][0]["code"] == "invalid_record"
    assert "evaluation_dataset_state/active" not in firestore.data
    assert service.calls == 0


def test_upload_and_run_has_a_dedicated_route() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    _service()
    try:
        response = TestClient(app).post(
            "/api/admin/evaluation-datasets/upload-and-run",
            content=json.dumps([_record()]),
            headers={"Content-Type": "application/json"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["run_id"] == "run-1"


def test_import_error_messages_and_logs_do_not_expose_evaluator_text(caplog) -> None:
    secret_question = "secret evaluator question"
    secret_passage = "secret evaluator passage"
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(FakeFirestore())
    _service()
    with caplog.at_level(logging.INFO):
        try:
            response = _post(TestClient(app), [_record(question=secret_question, supporting_passage=secret_passage, category="bad")])
        finally:
            app.dependency_overrides.clear()
    rendered = response.text + caplog.text
    assert response.status_code == status.HTTP_200_OK
    assert secret_question not in rendered
    assert secret_passage not in rendered


def test_active_dataset_exposes_only_ids_and_review_state() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    _service()
    try:
        client = TestClient(app)
        imported = _post(client, [_record(question="Sensitive evaluator question", supporting_passage="Sensitive evidence")])
        response = client.get("/api/admin/evaluation-datasets/active")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["generation_id"] == imported.json()["generation_id"]
    assert body["item_count"] == 1
    assert body["approved_gold_count"] == 0
    assert body["item_reviews"][0]["review_status"] == "candidate"
    assert "Sensitive evaluator question" not in response.text
    assert "Sensitive evidence" not in response.text


def test_versioned_dataset_revision_exposes_row_content_without_support_bindings() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    head = repository.stage_revision(
        "uploaded-corpus",
        1,
        (EvaluationRowV2(
            item_id="generated-row-1",
            origin="generated",
            question="What is metadata?",
            category="definitions",
            expected_outcome="answer",
            expected_answer_or_rubric="Explain metadata.",
            expected_refusal=False,
            source_spans=[{"span_id": "span-1", "block_id": "block-1", "physical_page": 3, "start_offset": 0, "end_offset": 8, "raw_text": "Metadata"}],
            relevant_child_ids=["internal-child-id"],
            parent_ids=["internal-parent-id"],
        ),),
    )
    repository.activate_revision(head, expected_revision=None)
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    _service()
    try:
        response = TestClient(app).get("/api/admin/evaluation-datasets/revision")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["revision"] == 1
    assert body["rows"][0]["question"] == "What is metadata?"
    assert body["rows"][0]["expected_answer_or_rubric"] == "Explain metadata."
    assert "source_spans" not in response.text
    assert "relevant_child_ids" not in response.text
    assert "internal-child-id" not in response.text
    assert "internal-parent-id" not in response.text


def test_active_dataset_and_review_require_admin_and_active_candidate() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    _service()
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "candidate-corpus"
    try:
        client = TestClient(app)
        assert client.get("/api/admin/evaluation-datasets/active").status_code == status.HTTP_404_NOT_FOUND
        imported = _post(client, [_record()]).json()
        generation_id = imported["generation_id"]
        active = client.get("/api/admin/evaluation-datasets/active").json()
        item_id = active["item_reviews"][0]["item_id"]
        approved = client.patch(
            f"/api/admin/evaluation-datasets/{generation_id}/items/{item_id}/review",
            json={"review_status": "approved"},
        )
        repeated = client.patch(
            f"/api/admin/evaluation-datasets/{generation_id}/items/{item_id}/review",
            json={"review_status": "rejected"},
        )
        active_after_review = client.get("/api/admin/evaluation-datasets/active")
        replacement = _post(client, [_record(question="A later generation")]).json()
        inactive = client.patch(
            f"/api/admin/evaluation-datasets/{generation_id}/items/{item_id}/review",
            json={"review_status": "rejected"},
        )
    finally:
        app.dependency_overrides.clear()

    assert approved.status_code == status.HTTP_200_OK
    assert approved.json()["review_status"] == "approved"
    assert repeated.status_code == status.HTTP_409_CONFLICT
    assert active_after_review.json()["approved_gold_count"] == 1
    assert replacement["generation_id"] != generation_id
    assert inactive.status_code == status.HTTP_409_CONFLICT


def test_binding_requires_real_candidate_chunk_and_persists_hash_before_approval() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    chunk_hash = sha256(b"Metadata describes data.").hexdigest()
    firestore.data["document_chunks_v2/chunk-real"] = {
        "text": "Metadata describes data.",
        "embedding": [0.0] * 768,
        "metadata": {
            "chunk_id": "chunk-real",
            "corpus_version_id": "candidate-corpus",
            "page": 1,
            "section": "Definitions",
            "chunk_ordinal": 0,
            "content_hash": chunk_hash,
        },
    }
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    _service()
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "candidate-corpus"
    try:
        client = TestClient(app)
        imported = _post(client, [_record()]).json()
        item_id = client.get("/api/admin/evaluation-datasets/active").json()["item_reviews"][0]["item_id"]
        url = f"/api/admin/evaluation-datasets/{imported['generation_id']}/items/{item_id}/binding"
        payload = {
            "candidate_snapshot_sha256": "a" * 64,
            "source_provenance_sha256": "b" * 64,
            "reviewed_chunks": [{"chunk_id": "chunk-real", "content_hash": chunk_hash}],
        }
        bound = client.put(url, json=payload)
        wrong_hash = client.put(url, json={**payload, "reviewed_chunks": [{"chunk_id": "chunk-real", "content_hash": "c" * 64}]})
        approved = client.patch(
            f"/api/admin/evaluation-datasets/{imported['generation_id']}/items/{item_id}/review",
            json={"review_status": "approved"},
        )
    finally:
        app.dependency_overrides.clear()

    assert bound.status_code == status.HTTP_200_OK, bound.text
    assert bound.json()["reviewed_chunk_ids"] == ["chunk-real"]
    assert wrong_hash.status_code == status.HTTP_409_CONFLICT
    stored = firestore.data[f"evaluation_generations/{imported['generation_id']}/annotations/{item_id}"]
    assert stored["relevant_chunk_ids"] == ["chunk-real"]
    assert stored["candidate_review"]["remap_status"] == "matched"
    assert approved.status_code == status.HTTP_200_OK


def test_expected_refusal_binding_has_no_supporting_chunk_and_can_be_approved() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    _service()
    app.dependency_overrides[uploaded_corpus_version_id] = lambda: "candidate-corpus"
    try:
        client = TestClient(app)
        imported = _post(client, [_record()]).json()
        item_id = client.get("/api/admin/evaluation-datasets/active").json()["item_reviews"][0]["item_id"]
        url = f"/api/admin/evaluation-datasets/{imported['generation_id']}/items/{item_id}/binding"
        response = client.put(
            url,
            json={
                "binding": {
                    "candidate_snapshot_sha256": "a" * 64,
                    "source_provenance_sha256": "b" * 64,
                },
                "expected_refusal": True,
            },
        )
        approved = client.patch(url.replace("/binding", "/review"), json={"review_status": "approved"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["expected_refusal"] is True
    assert approved.status_code == status.HTTP_200_OK
