import json
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import AnswerResponse, DocumentChunk, TelemetryEvent, validate_telemetry_payload
from app.contracts.api import (
    MAX_EVALUATION_IMPORT_BYTES,
    MAX_EVALUATION_IMPORT_ITEMS,
    ActiveEvaluationGeneration,
    ApiError,
    BatchAttestationRequest,
    BatchAttestationResult,
    BatchReviewStatus,
    DurableBatchAttestation,
    DurableExceptionDecision,
    DurableExceptionRecord,
    DurableRepresentativeSampleEntry,
    DurableReviewSummary,
    ExceptionRecord,
    ExplicitExceptionDecision,
    Citation,
    EvaluationImportResponse,
    EvaluationReportEligibility,
    ImportValidationError,
    ModelAttempt,
    QuestionRequest,
    QuotaStatus,
    RetrievalTrace,
    RepresentativeSampleEntry,
    ReviewDecisionMetadata,
    ReviewSummary,
)
from app.answering.pipeline import (
    DEFAULT_GENERATION_TIMEOUT_MS,
    DEFAULT_REQUEST_TIMEOUT_MS,
    MAX_EVIDENCE_PASSAGES,
    PARTIAL_EVIDENCE_THRESHOLD,
    STRONG_EVIDENCE_THRESHOLD,
)
from app.rag import LangChainRagSettings
from app.contracts.telemetry import TELEMETRY_FIELDS
from model.openrouter import OPENROUTER_FALLBACK_MODEL, OPENROUTER_PRIMARY_MODEL
from model.protocol import GenerationErrorCode


FIXTURE = json.loads(
    (Path(__file__).parents[2] / "documents/contracts/api-contract-fixture.json").read_text()
)


def quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=1,
        user_limit=100,
        global_used=1,
        global_limit=100,
        resets_at=datetime.now(timezone.utc),
    )


def trace() -> RetrievalTrace:
    return RetrievalTrace(retrieved_passages=[], selected_model="test-model")


def test_answer_contract_supports_all_outcomes_and_requires_safe_shape() -> None:
    answered = AnswerResponse(
        outcome="answer",
        answer_text="Grounded answer.",
        citations=[Citation(citation_id="c1", page=1, section="Intro")],
        trace=trace(),
        quota=quota(),
    )
    assert answered.outcome == "answer"

    qualified = AnswerResponse(outcome="qualified", answer_text="Limited answer.", trace=trace(), quota=quota())
    refused = AnswerResponse(outcome="refusal", answer_text="No relevant evidence.", trace=trace(), quota=quota())
    assert {qualified.outcome, refused.outcome} == {"qualified", "refusal"}

    with pytest.raises(ValidationError):
        AnswerResponse(outcome="answer", trace=trace(), quota=quota())


def test_document_chunk_requires_exactly_768_dimensions() -> None:
    valid = DocumentChunk(
        chunk_id="v1:1:0",
        corpus_version_id="v1",
        page=1,
        section="Intro",
        chunk_ordinal=0,
        content_hash="a" * 64,
        text="Evidence",
        embedding=[0.0] * FIXTURE["embeddingDimensions"],
    )
    assert len(valid.embedding) == 768

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id=valid.chunk_id,
            corpus_version_id=valid.corpus_version_id,
            page=valid.page,
            section=valid.section,
            chunk_ordinal=valid.chunk_ordinal,
            content_hash=valid.content_hash,
            text=valid.text,
            embedding=[0.0] * 767,
        )


def test_telemetry_is_allowlisted_and_rejects_interaction_content() -> None:
    event = TelemetryEvent(outcome="answer", duration_ms=12.5, request_count=1)
    assert set(event.model_dump(exclude_none=True)) <= set(TELEMETRY_FIELDS)

    with pytest.raises(ValueError):
        validate_telemetry_payload({"nested": {"question": "secret input"}})


def test_answer_response_cannot_be_forwarded_as_telemetry_payload() -> None:
    response = AnswerResponse(
        outcome="answer",
        answer_text="Do not persist this.",
        trace=trace(),
        quota=quota(),
    )

    with pytest.raises(ValueError, match="answer_text"):
        validate_telemetry_payload(response.model_dump())
    with pytest.raises(ValidationError):
        TelemetryEvent(question="secret input")  # type: ignore[call-arg]


def test_cross_language_fixture_matches_backend_contract_values() -> None:
    assert list(TELEMETRY_FIELDS) == FIXTURE["telemetryFields"]
    assert set(FIXTURE["outcomes"]) == {"answer", "qualified", "refusal"}


def test_frozen_public_schemas_match_typed_contracts() -> None:
    models = {
        "QuestionRequest": QuestionRequest,
        "ApiError": ApiError,
        "QuotaStatus": QuotaStatus,
        "Citation": Citation,
        "ModelAttempt": ModelAttempt,
        "RetrievalTrace": RetrievalTrace,
        "AnswerResponse": AnswerResponse,
    }

    for name, model in models.items():
        schema = model.model_json_schema()
        expected = FIXTURE["publicSchemas"][name]
        assert set(schema["properties"]) == set(expected["fields"])
        assert set(schema.get("required", [])) == set(expected["required"])
        assert schema["additionalProperties"] is False


def test_migration_baseline_matches_artifacts_and_effective_routes() -> None:
    baseline = FIXTURE["migrationBaseline"]
    candidate = baseline["compatibilityCandidate"]
    root = Path(__file__).parents[2]

    assert baseline["mergedIntegrationCommit"] == "50b9470836241cfcec0e9f60118c659f40d4fa17"
    assert candidate["lockSha256"] == sha256(
        (root / "backend/requirements-compat.lock").read_bytes()
    ).hexdigest()
    assert candidate["wheelSha256"] == sha256(
        (root / "backend/vendor/langchain-google-firestore/dist/langchain_google_firestore-0.5.0.post2-py3-none-any.whl").read_bytes()
    ).hexdigest()

    defaults = baseline["codeDefaults"]
    settings = LangChainRagSettings(project="contract-test")
    assert settings.location == defaults["location"]
    assert settings.vector_collection == defaults["vectorCollection"]
    assert settings.answer_provider == defaults["answerProvider"]
    assert DEFAULT_REQUEST_TIMEOUT_MS == defaults["answerRequestTimeoutMs"]
    assert DEFAULT_GENERATION_TIMEOUT_MS == defaults["answerGenerationTimeoutMs"]
    assert OPENROUTER_PRIMARY_MODEL == defaults["openrouterPrimaryModel"]
    assert OPENROUTER_FALLBACK_MODEL == defaults["openrouterFallbackModel"]

    effective = baseline["effectiveEnvironments"]
    assert effective["localCompose"] == {
        key: defaults[key]
        for key in (
            "answerProvider",
            "answerRequestTimeoutMs",
            "answerGenerationTimeoutMs",
            "openrouterPrimaryModel",
            "openrouterFallbackModel",
        )
    }
    assert effective["deliveryPreviewAndJobs"] == {
        **effective["localCompose"],
        "vectorCollection": defaults["vectorCollection"],
    }
    assert effective["vertexRoute"]["enabled"] is False
    assert "release gate" in effective["vertexRoute"]["releaseGate"].lower()

    delivery = (root / "scripts/deploy_cloud_run.sh").read_text()
    compose = (root / "docker-compose.yml").read_text()
    for value in effective["localCompose"].values():
        assert str(value) in compose
    for value in effective["deliveryPreviewAndJobs"].values():
        assert str(value) in delivery


def test_migration_baseline_freezes_policy_and_authored_parity_cases() -> None:
    baseline = FIXTURE["migrationBaseline"]
    policy = baseline["answeringPolicy"]
    assert policy["strongEvidenceThreshold"] == STRONG_EVIDENCE_THRESHOLD
    assert policy["partialEvidenceThreshold"] == PARTIAL_EVIDENCE_THRESHOLD
    assert policy["maxEvidencePassages"] == MAX_EVIDENCE_PASSAGES
    assert policy["eligibleFallbackErrors"] == [
        GenerationErrorCode.TIMEOUT.value,
        GenerationErrorCode.UNAVAILABLE.value,
        GenerationErrorCode.RATE_LIMITED.value,
    ]
    assert policy["maxModelAttempts"] == 2

    cases = baseline["authoredParityCases"]
    assert {case["id"] for case in cases} == {
        "strong-evidence",
        "partial-evidence",
        "absent-evidence",
        "eligible-fallback",
        "ineligible-fallback",
        "provenance-reconstruction",
    }
    assert all(case["productionContent"] is False for case in cases)

    documented = (Path(__file__).parents[2] / "documents/contracts/langchain-baseline.md").read_text()
    for value in (
        baseline["mergedIntegrationCommit"],
        baseline["compatibilityCandidate"]["commit"],
        baseline["compatibilityCandidate"]["compatibilityImage"],
        baseline["compatibilityCandidate"]["lockSha256"],
        baseline["compatibilityCandidate"]["wheelSha256"],
    ):
        assert value in documented


def test_evaluation_import_contract_handles_partial_and_zero_valid_results() -> None:
    partial = EvaluationImportResponse(
        status="accepted",
        generation_id="generation-2",
        submitted_count=3,
        imported_count=2,
        skipped_count=1,
        validation_errors=[ImportValidationError(index=1, code="invalid_category", message="Unsupported category.")],
    )
    assert partial.generation_id == "generation-2"


def test_batch_review_contracts_freeze_statuses_and_browser_safe_fields() -> None:
    statuses = FIXTURE["evaluation"]["batchReviewStatuses"]
    assert set(statuses) == {
        "pending", "eligible", "exception", "attested", "rejected", "stale", "inapplicable",
    }
    summary = ReviewSummary(
        status="eligible",
        eligible_count=2,
        exception_count=1,
        sampled_count=2,
        attested_count=0,
    )
    sample = RepresentativeSampleEntry(
        item_id="item-1",
        question="What is a data domain?",
        rubric="Defines the concept accurately.",
        support_excerpt="A bounded source excerpt.",
        page=7,
        section="Glossary",
        status="eligible",
    )
    exception = ExceptionRecord(
        item_id="item-2",
        question="What is a data domain?",
        rubric="Defines the concept accurately.",
        status="exception",
        reason="The supporting passage occurs more than once.",
    )
    assert summary.status in statuses
    assert sample.model_dump().keys() == {
        "item_id", "question", "rubric", "support_excerpt", "page", "section", "status", "decision",
    }
    assert set(exception.model_dump()).isdisjoint(set(FIXTURE["evaluation"]["browserReviewTechnicalFields"]))

    with pytest.raises(ValidationError):
        ReviewSummary(status="unknown", eligible_count=0, exception_count=0, sampled_count=0, attested_count=0)
    with pytest.raises(ValidationError):
        BatchAttestationRequest(decision="attested", candidate_snapshot_sha256="a" * 64)


def test_batch_review_contracts_bind_durable_records_and_audit_decisions() -> None:
    now = datetime.now(timezone.utc)
    metadata = ReviewDecisionMetadata(
        decision="attested", actor="admin-1", decided_at=now, decision_version="batch-review-v1"
    )
    decision = ExplicitExceptionDecision(item_id="item-2", decision="rejected", reason="Wrong support.")
    request = BatchAttestationRequest(decision="attested", exception_decisions=[decision])
    result = BatchAttestationResult(
        status="attested", decision=request.decision, exception_decisions=request.exception_decisions,
        decision_metadata=metadata,
    )
    common = {
        "generation_id": "generation-1",
        "run_id": "run-1",
        "candidate_snapshot_sha256": "a" * 64,
        "corpus_version_id": "corpus-1",
        "decision_version": "batch-review-v1",
    }
    summary = DurableReviewSummary(
        **common, status="attested", eligible_count=1, exception_count=1, sampled_item_ids=["item-1"],
        actor="admin-1", decided_at=now,
    )
    sample = DurableRepresentativeSampleEntry(
        **common, sample_identity="sample-1", item_id="item-1", candidate_chunk_ids=["chunk-1"],
        content_hashes=["b" * 64], provenance_digests=["c" * 64], status="attested",
    )
    exception = DurableExceptionRecord(
        **common, sample_identity="sample-2", item_id="item-2", candidate_chunk_ids=[],
        content_hashes=[], provenance_digests=[], status="exception", reason="Wrong support.",
    )
    durable_decision = DurableExceptionDecision(
        **common, sample_identity="sample-2", item_id="item-2", decision="rejected", actor="admin-1",
        decided_at=now, reason="Wrong support.",
    )
    attestation = DurableBatchAttestation(
        **common, sampled_item_ids=["item-1"], decision="attested", actor="admin-1", decided_at=now,
        exception_decisions=[durable_decision],
    )
    assert result.decision_metadata.actor == summary.actor == "admin-1"
    assert sample.status == "attested"
    assert exception.item_id == durable_decision.item_id == "item-2"
    assert attestation.exception_decisions[0].candidate_snapshot_sha256 == common["candidate_snapshot_sha256"]

    with pytest.raises(ValidationError):
        ExplicitExceptionDecision(item_id="item-2", decision="rejected")
    with pytest.raises(ValidationError):
        BatchAttestationRequest(
            decision="attested",
            exception_decisions=[
                ExplicitExceptionDecision(item_id="item-2", decision="attested"),
                ExplicitExceptionDecision(item_id="item-2", decision="attested"),
            ],
        )

    rejected = EvaluationImportResponse(
        status="rejected", submitted_count=2, imported_count=0, skipped_count=2
    )
    assert rejected.generation_id is None

    with pytest.raises(ValidationError, match="imported_count plus skipped_count"):
        EvaluationImportResponse(status="rejected", submitted_count=2, imported_count=1, skipped_count=0)


def test_release_evidence_requires_current_generation_and_gold_gate() -> None:
    current = EvaluationReportEligibility(
        generation_id="generation-2",
        eligibility="release_evidence",
        approved_gold_count=30,
        is_current_generation=True,
        is_superseded=False,
    )
    assert current.eligibility == "release_evidence"

    for count in (29, 51):
        with pytest.raises(ValidationError, match="release evidence"):
            EvaluationReportEligibility(
                generation_id="generation-2",
                eligibility="release_evidence",
                approved_gold_count=count,
                is_current_generation=True,
                is_superseded=False,
            )

    superseded = EvaluationReportEligibility(
        generation_id="generation-1",
        eligibility="exploratory",
        approved_gold_count=50,
        is_current_generation=False,
        is_superseded=True,
    )
    assert superseded.is_superseded


def test_evaluation_bounds_and_categories_match_fixture() -> None:
    evaluation = FIXTURE["evaluation"]
    assert MAX_EVALUATION_IMPORT_BYTES == evaluation["maxImportBytes"]
    assert MAX_EVALUATION_IMPORT_ITEMS == evaluation["maxImportItems"]
    assert ActiveEvaluationGeneration.model_fields["approved_gold_count"].metadata
