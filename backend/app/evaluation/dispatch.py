"""Durable, content-free dispatch of evaluation Cloud Run Jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Callable, Protocol
from uuid import uuid4

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore

from app.contracts import EvaluationDataset, EvaluationRun
from app.evaluation.job import EvaluationRunError


class EvaluationRunDispatcher(Protocol):
    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None: ...


class EvaluationDispatchError(EvaluationRunError):
    """A bounded Cloud Run provider diagnostic safe for an API response."""

    def __init__(self, status_code: int, provider_code: str, provider_message: str) -> None:
        super().__init__(provider_message)
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_message = provider_message


class EvaluationDispatchClaimStore:
    """Firestore claim records prevent duplicate job submissions after API restarts."""

    collection_id = "evaluation_run_dispatches"

    def __init__(self, client: Any) -> None:
        self._client = client
        self._claims = client.collection(self.collection_id)

    def claim(self, run_id: str) -> bool:
        try:
            self._claims.document(run_id).create(
                {"run_id": run_id, "requested_at": datetime.now(timezone.utc).isoformat()}
            )
            return True
        except AlreadyExists:
            return False

    def release(self, run_id: str, *, owner_id: str | None = None) -> None:
        reference = self._claims.document(run_id)
        if owner_id is None:
            reference.delete()
            return

        @firestore.transactional
        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() if snapshot.exists else {}
            if record.get("owner_id") == owner_id and record.get("state") in {"submitting", "submitted"}:
                transaction.delete(reference)

        apply(self._client.transaction())

    def claim_leased(
        self, run_id: str, *, lease_seconds: int = 120, now: datetime | None = None,
    ) -> str | None:
        """Acquire or recover an expired submission lease for resumable dispatch."""
        if lease_seconds < 1:
            raise ValueError("dispatch lease must be positive")
        instant = now or datetime.now(timezone.utc)
        reference = self._claims.document(run_id)
        owner_id = uuid4().hex

        @firestore.transactional
        def apply(transaction: Any) -> str | None:
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() if snapshot.exists else {}
            if record.get("state") == "submitted":
                return None
            lease_until = record.get("lease_until")
            if isinstance(lease_until, str):
                lease_until = datetime.fromisoformat(lease_until)
            if record.get("state") == "submitting" and isinstance(lease_until, datetime) and lease_until > instant:
                return None
            transaction.set(reference, {
                "run_id": run_id,
                "state": "submitting",
                "lease_until": (instant + timedelta(seconds=lease_seconds)).isoformat(),
                "owner_id": owner_id,
            })
            return owner_id

        return apply(self._client.transaction())

    def complete_claim(self, run_id: str, *, owner_id: str) -> None:
        reference = self._claims.document(run_id)

        @firestore.transactional
        def apply(transaction: Any) -> None:
            snapshot = reference.get(transaction=transaction)
            record = snapshot.to_dict() if snapshot.exists else {}
            if record.get("owner_id") == owner_id and record.get("state") == "submitting":
                transaction.update(reference, {"state": "submitted", "lease_until": None})

        apply(self._client.transaction())


class CloudRunJobExecutor:
    """Start one Cloud Run execution with only the opaque run identifier."""

    def __init__(self, *, project: str, location: str, job_name: str) -> None:
        self.project = _non_empty(project, "project")
        self.location = _non_empty(location, "location")
        self.job_name = _non_empty(job_name, "job_name")

    def __call__(self, run_id: str) -> None:
        self.dispatch_overrides({"EVALUATION_RUN_ID": run_id})

    def dispatch_overrides(self, environment: dict[str, str]) -> None:
        """Start one execution with a bounded set of server-created variables."""
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        response = AuthorizedSession(credentials).post(
            "https://run.googleapis.com/v2/projects/"
            f"{self.project}/locations/{self.location}/jobs/{self.job_name}:run",
            json={
                "overrides": {
                    "containerOverrides": [
                        {"env": [{"name": name, "value": value} for name, value in sorted(environment.items())]}
                    ]
                }
            },
            timeout=15,
        )
        if not response.ok:
            status_code, provider_code, provider_message = _safe_provider_diagnostic(response)
            raise EvaluationDispatchError(status_code, provider_code, provider_message)


class CloudRunCandidateGenerationDispatcher:
    """Submit one durable generation task per corpus and generator fingerprint."""

    def __init__(self, claims: EvaluationDispatchClaimStore, executor: CloudRunJobExecutor) -> None:
        self.claims = claims
        self.executor = executor

    @staticmethod
    def operation_key(corpus_version_id: str, generator_fingerprint: str) -> str:
        return f"candidate-generation:{corpus_version_id}:{generator_fingerprint}"

    @staticmethod
    def claim_id(operation_key: str) -> str:
        return "candidate-generation-" + sha256(operation_key.encode("utf-8")).hexdigest()

    def dispatch(self, *, attempt_id: str, corpus_version_id: str, generator_fingerprint: str) -> bool:
        operation_key = self.operation_key(corpus_version_id, generator_fingerprint)
        claim_id = self.claim_id(operation_key)
        owner_id = self.claims.claim_leased(claim_id)
        if owner_id is None:
            return False
        try:
            self.executor.dispatch_overrides({
                "EVALUATION_TASK": "prepare_candidates",
                "CORPUS_STAGING_ATTEMPT_ID": attempt_id,
                "EVALUATION_CORPUS_VERSION_ID": corpus_version_id,
                "EVALUATION_GENERATOR_FINGERPRINT": generator_fingerprint,
                "EVALUATION_DISPATCH_OWNER_ID": owner_id,
            })
        except Exception:
            self.claims.release(claim_id, owner_id=owner_id)
            raise
        self.claims.complete_claim(claim_id, owner_id=owner_id)
        return True


class CloudRunEvaluationJobDispatcher:
    """Claim then dispatch exactly one durable execution request per run."""

    def __init__(self, claims: EvaluationDispatchClaimStore, execute: Callable[[str], None]) -> None:
        self._claims = claims
        self._execute = execute

    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None:
        # The dataset is deliberately not passed to Cloud Run: its run ID binds
        # immutable identifiers without exposing evaluator-authored content.
        del dataset
        if not self._claims.claim(run.run_id):
            return
        try:
            self._execute(run.run_id)
        except Exception:
            # A rejected request must remain retryable. Successful requests
            # retain their claim across API process restarts.
            self._claims.release(run.run_id)
            raise


def _non_empty(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


_SAFE_PROVIDER_CODES = frozenset({
    "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
    "UNAUTHENTICATED", "UNAVAILABLE", "INTERNAL", "CLOUD_RUN_DISPATCH_FAILED",
})
_STATUS_DIAGNOSTICS = {
    400: ("INVALID_ARGUMENT", "Cloud Run rejected the job override request."),
    401: ("UNAUTHENTICATED", "Cloud Run rejected the caller authentication."),
    403: ("PERMISSION_DENIED", "Cloud Run rejected the caller permissions."),
    404: ("NOT_FOUND", "The evaluation Cloud Run Job was not found."),
    429: ("RESOURCE_EXHAUSTED", "Cloud Run rejected the request due to quota or rate limiting."),
}


def _safe_provider_diagnostic(response: Any) -> tuple[int, str, str]:
    status_code = int(getattr(response, "status_code", 0) or 0)
    fallback_code, fallback_message = _STATUS_DIAGNOSTICS.get(
        status_code,
        ("UNAVAILABLE" if status_code >= 500 else "CLOUD_RUN_DISPATCH_FAILED", "Cloud Run did not accept the evaluation job dispatch."),
    )
    try:
        payload = response.json()
    except (AttributeError, TypeError, ValueError):
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    raw_code = (error.get("status") or error.get("code")) if isinstance(error, dict) else None
    provider_code = raw_code if isinstance(raw_code, str) and raw_code in _SAFE_PROVIDER_CODES else fallback_code
    return status_code, provider_code, fallback_message
