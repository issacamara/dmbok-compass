"""Durable, exact-output human assessments for evaluator-authored runs."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Protocol

from google.api_core.exceptions import AlreadyExists
from pydantic import Field

from app.contracts.api import CandidateIdentity, ContractModel, UserProfile


class AssessmentError(ValueError):
    """Base error for assessment validation and persistence failures."""


class AssessmentConflictError(AssessmentError):
    """The submitted assessment conflicts with immutable run/output state."""


class AssessmentNotFoundError(AssessmentError):
    """The requested exact evaluator output is not available."""


class RunItemAssessment(ContractModel):
    """The reviewer decision for one immutable candidate output."""

    generation_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    scorer_version: str = Field(min_length=1, max_length=128)
    output_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    claim_support: bool | None = None
    citation_support: bool | None = None
    rubric_passed: bool | None = None
    refusal_judgment: bool | None = None

    @property
    def complete(self) -> bool:
        return (
            self.claim_support is not None
            and self.citation_support is not None
            and self.rubric_passed is not None
            and self.refusal_judgment is not None
        )


class RunItemAssessmentRecord(RunItemAssessment):
    reviewer_actor: str = Field(min_length=1, max_length=128)
    reviewed_at: datetime
    assessment_id: str = Field(min_length=1, max_length=256)
    candidate_identity: CandidateIdentity


class OutputAudit(Protocol):
    evaluation_run_id: str
    evaluation_item_id: str
    candidate_id: str
    candidate_identity: CandidateIdentity
    scorer_version: str
    output_digest: str


class OutputStore(Protocol):
    def get_audit(self, attempt_id: str) -> OutputAudit | None: ...


class AssessmentRepository(Protocol):
    def save(
        self,
        assessment: RunItemAssessment,
        *,
        actor: UserProfile,
        candidate_identity: CandidateIdentity,
        reviewed_at: datetime,
    ) -> RunItemAssessmentRecord: ...


class InMemoryAssessmentRepository:
    """Restart-safe seam used by endpoint tests and local fixtures."""

    def __init__(self) -> None:
        self.records: dict[str, RunItemAssessmentRecord] = {}

    def save(self, assessment: RunItemAssessment, *, actor: UserProfile, candidate_identity: CandidateIdentity, reviewed_at: datetime) -> RunItemAssessmentRecord:
        key = assessment_key(assessment)
        existing = self.records.get(key)
        if existing is not None:
            if existing.model_dump(exclude={"reviewer_actor", "reviewed_at"}) != {
                **assessment.model_dump(), "candidate_identity": candidate_identity.model_dump(mode="json"), "assessment_id": key,
            }:
                raise AssessmentConflictError("assessment identity is already bound to another decision")
            return existing
        record = RunItemAssessmentRecord(
            **assessment.model_dump(), assessment_id=key, candidate_identity=candidate_identity,
            reviewer_actor=actor.user_id, reviewed_at=reviewed_at,
        )
        self.records[key] = record
        return record


class FirestoreAssessmentRepository:
    """Immutable assessment records keyed by exact run/item/output digest."""

    collection_id = "evaluation_run_item_assessments"

    def __init__(self, client: Any) -> None:
        self.assessments = client.collection(self.collection_id)

    def save(self, assessment: RunItemAssessment, *, actor: UserProfile, candidate_identity: CandidateIdentity, reviewed_at: datetime) -> RunItemAssessmentRecord:
        key = assessment_key(assessment)
        record = RunItemAssessmentRecord(
            **assessment.model_dump(), assessment_id=key, candidate_identity=candidate_identity,
            reviewer_actor=actor.user_id, reviewed_at=reviewed_at,
        )
        reference = self.assessments.document(key)
        payload = record.model_dump(mode="json")
        try:
            reference.create(payload)
        except AlreadyExists:
            snapshot = reference.get()
            if not snapshot.exists:
                raise AssessmentError("assessment storage could not be read after a conflict")
            existing = RunItemAssessmentRecord.model_validate(snapshot.to_dict())
            if existing.model_dump(exclude={"reviewer_actor", "reviewed_at"}) != record.model_dump(exclude={"reviewer_actor", "reviewed_at"}):
                raise AssessmentConflictError("assessment identity is already bound to another decision")
            return existing
        return record


def assessment_key(assessment: RunItemAssessment) -> str:
    material = "|".join((assessment.generation_id, assessment.attempt_id, assessment.candidate_id, assessment.output_digest))
    return "assessment-" + sha256(material.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
