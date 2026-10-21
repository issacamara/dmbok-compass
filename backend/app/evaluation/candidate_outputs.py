"""Admin-restricted, bounded-retention evaluator candidate output storage.

This repository is deliberately separate from production answer and telemetry
storage.  Raw output is temporary review material; the audit record is the
durable, content-free result of that review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Literal, Protocol

from google.api_core.exceptions import AlreadyExists

from app.contracts import CandidateIdentity, UserProfile
from app.identity import require_admin


POLICY_ID = "evaluator-output-retention-v1"
MAX_RAW_OUTPUT_AGE = timedelta(days=30)
EvaluatorOutputSource = Literal["administrator_authored_evaluation_question"]


class CandidateOutputError(ValueError):
    """Base error for evaluator-output persistence failures."""


class CandidateOutputAuthorizationError(CandidateOutputError):
    """Raised without revealing whether a protected output exists."""


class CandidateOutputBindingError(CandidateOutputError):
    """Raised when an output is not bound to the supplied evaluation identity."""


class CandidateOutputSourceError(CandidateOutputError):
    """Raised when production interaction material reaches this repository."""


class CandidateOutputAlreadyExists(CandidateOutputError):
    """Raised when an immutable attempt identity is reused."""


class _Snapshot(Protocol):
    exists: bool

    def to_dict(self) -> dict[str, Any]: ...


class _Document(Protocol):
    def get(self) -> _Snapshot: ...

    def create(self, data: dict[str, Any]) -> Any: ...

    def set(self, data: dict[str, Any]) -> Any: ...

    def delete(self) -> Any: ...


class _Collection(Protocol):
    def document(self, document_id: str) -> _Document: ...

    def stream(self) -> list[_Snapshot]: ...


class _FirestoreClient(Protocol):
    def collection(self, collection_id: str) -> _Collection: ...


@dataclass(frozen=True)
class CandidateOutputAudit:
    evaluation_run_id: str
    evaluation_item_id: str
    candidate_id: str
    attempt_id: str
    candidate_identity: CandidateIdentity
    scorer_version: str
    output_digest: str
    assessment: dict[str, Any] | None
    assessment_status: Literal["pending", "reviewed"]
    reviewer_actor: str | None
    reviewed_at: datetime | None
    policy_id: str = POLICY_ID


@dataclass(frozen=True)
class RetentionCleanupResult:
    policy_id: str
    checked_at: datetime
    deleted_attempt_ids: tuple[str, ...]
    recovery: bool


class FirestoreCandidateOutputStore:
    """Persist temporary raw outputs and durable content-free audit records."""

    raw_collection_id = "evaluation_candidate_outputs"
    audit_collection_id = "evaluation_candidate_output_audit"

    def __init__(self, client: _FirestoreClient) -> None:
        self.raw = client.collection(self.raw_collection_id)
        self.audit = client.collection(self.audit_collection_id)

    def put(
        self,
        *,
        actor: UserProfile,
        evaluation_run_id: str,
        evaluation_item_id: str,
        candidate_id: str,
        attempt_id: str,
        candidate_identity: CandidateIdentity,
        output: str,
        generated_at: datetime,
        source: EvaluatorOutputSource,
        scorer_version: str,
        assessment: dict[str, Any] | None = None,
    ) -> CandidateOutputAudit:
        _authorize(actor)
        _validate_binding(
            evaluation_run_id, evaluation_item_id, candidate_id, attempt_id,
            candidate_identity, scorer_version, source, output,
        )
        generated_at = _utc(generated_at)
        # The attempt identity is the document key.  The tuple is still
        # persisted and validated so an attempt cannot be rebound silently.
        key = attempt_id
        audit = _audit(
            evaluation_run_id=evaluation_run_id,
            evaluation_item_id=evaluation_item_id,
            candidate_id=candidate_id,
            attempt_id=attempt_id,
            candidate_identity=candidate_identity,
            scorer_version=scorer_version,
            output_digest=_digest(output),
            assessment=assessment,
            assessment_status="reviewed" if assessment is not None else "pending",
            reviewer_actor=actor.user_id if assessment is not None else None,
            reviewed_at=generated_at if assessment is not None else None,
        )
        raw = {
            "evaluation_run_id": evaluation_run_id,
            "evaluation_item_id": evaluation_item_id,
            "candidate_id": candidate_id,
            "attempt_id": attempt_id,
            "candidate_identity": candidate_identity.model_dump(mode="json"),
            "scorer_version": scorer_version,
            "output": output,
            "output_digest": audit.output_digest,
            "generated_at": generated_at.isoformat(),
            "expires_at": (generated_at + MAX_RAW_OUTPUT_AGE).isoformat(),
            "policy_id": POLICY_ID,
            "source": source,
        }
        try:
            self.raw.document(key).create(raw)
            self.audit.document(key).create(_audit_payload(audit))
        except AlreadyExists as exc:
            raise CandidateOutputAlreadyExists("attempt_id is already bound to an evaluator output") from exc
        return audit

    def get_raw(self, *, actor: UserProfile, attempt_id: str) -> str | None:
        _authorize(actor)
        snapshot = self.raw.document(attempt_id).get()
        if not snapshot.exists:
            return None
        payload = snapshot.to_dict()
        if payload.get("policy_id") != POLICY_ID:
            return None
        return payload.get("output") if isinstance(payload.get("output"), str) else None

    def finalize(self, *, actor: UserProfile, attempt_id: str, assessment: dict[str, Any] | None = None) -> CandidateOutputAudit | None:
        _authorize(actor)
        raw_ref = self.raw.document(attempt_id)
        snapshot = raw_ref.get()
        if not snapshot.exists:
            return self._audit_for(attempt_id)
        payload = snapshot.to_dict()
        if payload.get("policy_id") != POLICY_ID:
            raise CandidateOutputBindingError("stored output has an invalid policy")
        audit = self._audit_for(attempt_id)
        if audit is None:
            audit = _audit_from_raw(payload, assessment=assessment, reviewer_actor=actor.user_id)
            self.audit.document(attempt_id).set(_audit_payload(audit))
        elif assessment is not None and audit.assessment is None:
            audit = _audit_from_raw(payload, assessment=assessment, reviewer_actor=actor.user_id)
            self.audit.document(attempt_id).set(_audit_payload(audit))
        raw_ref.delete()
        return audit

    def cleanup_expired(self, *, actor: UserProfile, now: datetime | None = None, recovery: bool = False) -> RetentionCleanupResult:
        _authorize(actor)
        checked_at = _utc(now or datetime.now(timezone.utc))
        deleted: list[str] = []
        for snapshot in self.raw.stream():
            payload = snapshot.to_dict()
            attempt_id = payload.get("attempt_id")
            expires_at = _parse_datetime(payload.get("expires_at"))
            if not isinstance(attempt_id, str) or payload.get("policy_id") != POLICY_ID:
                continue
            if expires_at is not None and expires_at <= checked_at:
                self.finalize(actor=actor, attempt_id=attempt_id)
                deleted.append(attempt_id)
        return RetentionCleanupResult(POLICY_ID, checked_at, tuple(deleted), recovery)

    def _audit_for(self, attempt_id: str) -> CandidateOutputAudit | None:
        snapshot = self.audit.document(attempt_id).get()
        return _audit_from_payload(snapshot.to_dict()) if snapshot.exists else None


def _authorize(actor: UserProfile) -> None:
    try:
        require_admin(actor)
    except Exception as exc:
        raise CandidateOutputAuthorizationError("administrator access is required") from exc


def _validate_binding(
    evaluation_run_id: str,
    evaluation_item_id: str,
    candidate_id: str,
    attempt_id: str,
    candidate_identity: CandidateIdentity,
    scorer_version: str,
    source: EvaluatorOutputSource,
    output: str,
) -> None:
    if not all(isinstance(value, str) and value.strip() for value in (evaluation_run_id, evaluation_item_id, candidate_id, attempt_id, scorer_version)):
        raise CandidateOutputBindingError("run, item, candidate, attempt, and scorer identities are required")
    if source != "administrator_authored_evaluation_question":
        raise CandidateOutputSourceError("production interactions cannot be stored as evaluator outputs")
    if not output:
        raise CandidateOutputBindingError("candidate output must not be empty")
    if candidate_identity.scorer_version != scorer_version:
        raise CandidateOutputBindingError("candidate identity and scorer version do not match")


def _digest(output: str) -> str:
    return sha256(output.encode("utf-8")).hexdigest()


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def _audit(**kwargs: Any) -> CandidateOutputAudit:
    return CandidateOutputAudit(**kwargs)


def _audit_from_raw(payload: dict[str, Any], *, assessment: dict[str, Any] | None = None, reviewer_actor: str | None = None) -> CandidateOutputAudit:
    identity = CandidateIdentity.model_validate(payload["candidate_identity"])
    reviewed_at = _utc(datetime.now(timezone.utc)) if assessment is not None else None
    return _audit(
        evaluation_run_id=payload["evaluation_run_id"], evaluation_item_id=payload["evaluation_item_id"],
        candidate_id=payload["candidate_id"], attempt_id=payload["attempt_id"], candidate_identity=identity,
        scorer_version=payload["scorer_version"], output_digest=payload["output_digest"], assessment=assessment,
        assessment_status="reviewed" if assessment is not None else "pending",
        reviewer_actor=reviewer_actor, reviewed_at=reviewed_at,
    )


def _audit_from_payload(payload: dict[str, Any]) -> CandidateOutputAudit:
    return _audit(
        evaluation_run_id=payload["evaluation_run_id"], evaluation_item_id=payload["evaluation_item_id"],
        candidate_id=payload["candidate_id"], attempt_id=payload["attempt_id"],
        candidate_identity=CandidateIdentity.model_validate(payload["candidate_identity"]),
        scorer_version=payload["scorer_version"], output_digest=payload["output_digest"],
        assessment=payload.get("assessment"), assessment_status=payload["assessment_status"],
        reviewer_actor=payload.get("reviewer_actor"), reviewed_at=_parse_datetime(payload.get("reviewed_at")),
        policy_id=payload.get("policy_id", ""),
    )


def _audit_payload(audit: CandidateOutputAudit) -> dict[str, Any]:
    return {
        "evaluation_run_id": audit.evaluation_run_id, "evaluation_item_id": audit.evaluation_item_id,
        "candidate_id": audit.candidate_id, "attempt_id": audit.attempt_id,
        "candidate_identity": audit.candidate_identity.model_dump(mode="json"),
        "scorer_version": audit.scorer_version, "output_digest": audit.output_digest,
        "assessment": audit.assessment, "assessment_status": audit.assessment_status,
        "reviewer_actor": audit.reviewer_actor,
        "reviewed_at": audit.reviewed_at.isoformat() if audit.reviewed_at else None,
        "policy_id": audit.policy_id,
    }
