"""Typed public API and durable-domain contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Outcome = Literal["answer", "qualified", "refusal"]
QuestionCategory = Literal["definitions", "explanations", "comparisons", "study", "scenarios"]
ImportStatus = Literal["accepted", "rejected"]
ReviewStatus = Literal["candidate", "approved", "rejected"]
RunEligibility = Literal["exploratory", "release_evidence"]
RunStatus = Literal["queued", "running", "completed", "failed"]

# The upload contract is intentionally shared with the web client. Keep these
# values in sync with documents/contracts/api-contract-fixture.json.
MAX_EVALUATION_IMPORT_BYTES = 1_048_576
MAX_EVALUATION_IMPORT_ITEMS = 500
MIN_APPROVED_GOLD_ITEMS = 30
MAX_APPROVED_GOLD_ITEMS = 50


class ApiError(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False
    status: int = Field(default=400, ge=400, le=599)


class QuotaStatus(ContractModel):
    user_used: int = Field(ge=0)
    user_limit: int = Field(ge=0)
    global_used: int = Field(ge=0)
    global_limit: int = Field(ge=0)
    resets_at: datetime


class Citation(ContractModel):
    citation_id: str = Field(min_length=1, max_length=128)
    page: int = Field(ge=1)
    section: str = Field(min_length=1, max_length=500)
    excerpt: str = Field(min_length=1, max_length=5000)


class RetrievedPassage(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    page: int = Field(ge=1)
    section: str = Field(min_length=1, max_length=500)
    excerpt: str = Field(min_length=1, max_length=5000)
    relevance_score: float | None = None


class QuestionRequest(ContractModel):
    question: str = Field(min_length=1, max_length=2000)


class ModelAttempt(ContractModel):
    model: str = Field(min_length=1, max_length=128)
    outcome: Literal[
        "success", "invalid_output", "timeout", "unavailable", "rate_limited",
        "authentication_failed", "request_rejected", "unknown",
    ]
    status_code: int | None = Field(default=None, ge=100, le=599)


class RetrievalTrace(ContractModel):
    retrieved_passages: list[RetrievedPassage] = Field(max_length=5)
    selected_model: str = Field(min_length=1, max_length=128)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    model_attempts: list[ModelAttempt] = Field(default_factory=list, max_length=2)
    model_output: str | None = Field(default=None, max_length=20_000)


class AnswerResponse(ContractModel):
    outcome: Outcome
    answer_text: str | None = Field(default=None, max_length=20000)
    synthesis: bool = False
    citations: list[Citation] = Field(default_factory=list)
    trace: RetrievalTrace
    quota: QuotaStatus
    error: ApiError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "AnswerResponse":
        if self.outcome in {"answer", "qualified"} and not self.answer_text:
            raise ValueError("answer_text is required for answer and qualified outcomes")
        if self.outcome == "refusal" and self.citations:
            raise ValueError("refusal responses cannot include citations")
        if self.outcome == "refusal" and self.error is None and not self.answer_text:
            raise ValueError("refusal responses require answer_text or error")
        return self


class UserProfile(ContractModel):
    user_id: str = Field(min_length=1, max_length=128)
    email: str = Field(min_length=3, max_length=320)
    username: str = Field(min_length=1, max_length=80)
    approval_state: Literal["pending", "approved", "rejected", "deactivated"]
    role: Literal["user", "admin"] = "user"
    email_verified: bool = False


class QuotaPolicy(ContractModel):
    per_user_daily_limit: int = Field(default=100, ge=0)
    global_daily_limit: int = Field(default=100, ge=0)


class AggregateMetric(ContractModel):
    metric_name: str = Field(min_length=1, max_length=128)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class EvaluationMetric(AggregateMetric):
    """Content-free release-gate result for one evaluated metric."""

    threshold: float = Field(ge=0, le=100)
    passed: bool


class ImportValidationError(ContractModel):
    """One content-free validation result for an imported array element."""

    index: int = Field(ge=0)
    code: Literal["invalid_record", "missing_question", "missing_expected_answer_or_rubric", "missing_supporting_passage", "invalid_category"]
    message: str = Field(min_length=1, max_length=500)


class EvaluationImportResponse(ContractModel):
    """The complete outcome of a bounded administrator JSON import."""

    status: ImportStatus
    generation_id: str | None = Field(default=None, min_length=1, max_length=128)
    submitted_count: int = Field(ge=0, le=MAX_EVALUATION_IMPORT_ITEMS)
    imported_count: int = Field(ge=0, le=MAX_EVALUATION_IMPORT_ITEMS)
    skipped_count: int = Field(ge=0, le=MAX_EVALUATION_IMPORT_ITEMS)
    validation_errors: list[ImportValidationError] = Field(default_factory=list, max_length=MAX_EVALUATION_IMPORT_ITEMS)

    @model_validator(mode="after")
    def validate_counts_and_generation(self) -> "EvaluationImportResponse":
        if self.imported_count + self.skipped_count != self.submitted_count:
            raise ValueError("imported_count plus skipped_count must equal submitted_count")
        if self.status == "accepted" and (not self.generation_id or self.imported_count == 0):
            raise ValueError("accepted imports require a generation_id and at least one imported item")
        if self.status == "rejected" and self.generation_id is not None:
            raise ValueError("rejected imports cannot activate a generation")
        return self


class ActiveEvaluationGeneration(ContractModel):
    generation_id: str = Field(min_length=1, max_length=128)
    item_count: int = Field(ge=1, le=MAX_EVALUATION_IMPORT_ITEMS)
    approved_gold_count: int = Field(ge=0, le=MAX_EVALUATION_IMPORT_ITEMS)
    status: Literal["active"] = "active"


class ItemReview(ContractModel):
    generation_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    review_status: ReviewStatus


class EvaluationRunSelection(ContractModel):
    """Launch input. The server resolves the active generation, never the client."""

    item_ids: list[str] | None = Field(default=None, max_length=MAX_EVALUATION_IMPORT_ITEMS)


class EvaluationReportEligibility(ContractModel):
    generation_id: str = Field(min_length=1, max_length=128)
    eligibility: RunEligibility
    approved_gold_count: int = Field(ge=0, le=MAX_EVALUATION_IMPORT_ITEMS)
    is_current_generation: bool
    is_superseded: bool

    @model_validator(mode="after")
    def validate_visibility(self) -> "EvaluationReportEligibility":
        if self.is_current_generation == self.is_superseded:
            raise ValueError("current-generation and superseded flags must be opposites")
        if self.eligibility == "release_evidence" and not (
            self.is_current_generation and MIN_APPROVED_GOLD_ITEMS <= self.approved_gold_count <= MAX_APPROVED_GOLD_ITEMS
        ):
            raise ValueError("release evidence requires the current generation and 30–50 approved gold items")
        return self


class CorpusVersion(ContractModel):
    version_id: str = Field(min_length=1, max_length=128)
    source_uri: str = Field(min_length=1, max_length=2048)
    embedding_model: str = Field(min_length=1, max_length=128)
    embedding_dimensions: Literal[768] = 768
    status: Literal["staged", "validated", "active", "retired"]
    created_at: datetime


class DocumentChunk(ContractModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    corpus_version_id: str = Field(min_length=1, max_length=128)
    page: int = Field(ge=1)
    section: str = Field(min_length=1, max_length=500)
    chunk_ordinal: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str = Field(min_length=1, max_length=20000)
    embedding: list[float] = Field(min_length=768, max_length=768)


class EvaluationItem(ContractModel):
    item_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    category: QuestionCategory = "definitions"


class GoldAnnotation(ContractModel):
    item_id: str = Field(min_length=1, max_length=128)
    expected_answer_or_rubric: str = Field(min_length=1, max_length=10000)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    review_status: Literal["candidate", "approved", "rejected"]


class EvaluationDataset(ContractModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    items: list[EvaluationItem] = Field(min_length=1)
    gold_annotations: list[GoldAnnotation] = Field(default_factory=list)


class EvaluationItemResult(ContractModel):
    """Content-free outcome for one evaluator-authored item."""

    item_id: str = Field(min_length=1, max_length=128)
    retrieval_success: bool
    grounded: bool
    citation_correct: bool
    answer_quality: bool
    refusal_correct: bool
    response_time_ms: float = Field(ge=0)


class EvaluationRun(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    corpus_version_id: str = Field(min_length=1, max_length=128)
    status: RunStatus
    metrics: list[EvaluationMetric] = Field(default_factory=list)
    item_results: list[EvaluationItemResult] = Field(default_factory=list, max_length=10000)
    selected_item_ids: list[str] = Field(default_factory=list, max_length=10000)
    configuration_version_id: str = Field(min_length=1, max_length=128)
    model_version_id: str = Field(min_length=1, max_length=128)
    error: str | None = Field(default=None, max_length=500)
    report_eligibility: EvaluationReportEligibility | None = None


class ReleaseDecision(ContractModel):
    release_id: str = Field(min_length=1, max_length=128)
    evaluation_run_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    corpus_version_id: str = Field(min_length=1, max_length=128)
    configuration_version_id: str = Field(min_length=1, max_length=128)
    provider_version_id: str = Field(min_length=1, max_length=128)
    model_version_id: str = Field(min_length=1, max_length=128)
    scorer_version_id: str = Field(min_length=1, max_length=128)
    gate_report_ids: list[str] = Field(min_length=1, max_length=100)
    decision: Literal["pending", "approved", "rejected"]
    rationale: str = Field(min_length=1, max_length=5000)
    exception_approved: bool = False
    exception_rationale: str | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def validate_exception(self) -> "ReleaseDecision":
        if self.exception_approved and not self.exception_rationale:
            raise ValueError("an approved exception requires an exception rationale")
        if not self.exception_approved and self.exception_rationale:
            raise ValueError("exception rationale requires an approved exception")
        return self


API_ROUTES = {
    "registration": "/api/registration",
    "registration_status": "/api/registration/status",
    "current_user": "/api/me",
    "answer": "/api/questions",
    "quota": "/api/quota",
    "evaluation_runs": "/api/evaluations",
    "evaluation_import": "/api/admin/evaluation-datasets/import",
    "active_evaluation_generation": "/api/admin/evaluation-datasets/active",
    "release_decisions": "/api/releases",
    "evaluation_datasets": "/api/admin/evaluation-datasets",
}

ADMIN_ROUTES = {
    "users": "/api/admin/users",
    "configuration": "/api/admin/configuration",
    "aggregate_metrics": "/api/admin/metrics",
    "release_decisions": "/api/releases",
}
