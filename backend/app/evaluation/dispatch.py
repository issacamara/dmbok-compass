"""Durable, content-free dispatch of evaluation Cloud Run Jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from google.api_core.exceptions import AlreadyExists

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
        self._claims = client.collection(self.collection_id)

    def claim(self, run_id: str) -> bool:
        try:
            self._claims.document(run_id).create(
                {"run_id": run_id, "requested_at": datetime.now(timezone.utc).isoformat()}
            )
            return True
        except AlreadyExists:
            return False

    def release(self, run_id: str) -> None:
        self._claims.document(run_id).delete()


class CloudRunJobExecutor:
    """Start one Cloud Run execution with only the opaque run identifier."""

    def __init__(self, *, project: str, location: str, job_name: str) -> None:
        self.project = _non_empty(project, "project")
        self.location = _non_empty(location, "location")
        self.job_name = _non_empty(job_name, "job_name")

    def __call__(self, run_id: str) -> None:
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
                        {"env": [{"name": "EVALUATION_RUN_ID", "value": run_id}]}
                    ]
                }
            },
            timeout=15,
        )
        if not response.ok:
            status_code, provider_code, provider_message = _safe_provider_diagnostic(response)
            raise EvaluationDispatchError(status_code, provider_code, provider_message)


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
