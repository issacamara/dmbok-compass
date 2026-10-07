"""Typed public API and durable-domain contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


Outcome = Literal["answer", "qualified", "refusal"]
QuestionCategory = Literal["definitions", "explanations", "comparisons", "study", "scenarios"]


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


class RetrievalTrace(ContractModel):
    retrieved_passages: list[RetrievedPassage] = Field(max_length=5)
    selected_model: str = Field(min_length=1, max_length=128)
    timings_ms: dict[str, float] = Field(default_factory=dict)


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
    status: Literal["queued", "running", "completed", "failed"]
    metrics: list[AggregateMetric] = Field(default_factory=list)
    item_results: list[EvaluationItemResult] = Field(default_factory=list, max_length=10000)
    selected_item_ids: list[str] = Field(default_factory=list, max_length=10000)
    configuration_version_id: str = Field(min_length=1, max_length=128)
    model_version_id: str = Field(min_length=1, max_length=128)
    error: str | None = Field(default=None, max_length=500)


class ReleaseDecision(ContractModel):
    release_id: str = Field(min_length=1, max_length=128)
    evaluation_run_id: str = Field(min_length=1, max_length=128)
    decision: Literal["pending", "approved", "rejected"]
    rationale: str = Field(min_length=1, max_length=5000)


API_ROUTES = {
    "registration": "/api/registration",
    "registration_status": "/api/registration/status",
    "current_user": "/api/me",
    "answer": "/api/questions",
    "quota": "/api/quota",
    "evaluation_runs": "/api/evaluations",
    "release_decisions": "/api/releases",
}

ADMIN_ROUTES = {
    "users": "/api/admin/users",
    "configuration": "/api/admin/configuration",
    "aggregate_metrics": "/api/admin/metrics",
}
