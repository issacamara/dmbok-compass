"""Durable, content-free dispatch of evaluation Cloud Run Jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from google.api_core.exceptions import AlreadyExists

from app.contracts import EvaluationDataset, EvaluationRun
from app.evaluation.job import EvaluationRunError


class EvaluationRunDispatcher(Protocol):
    def dispatch(self, run: EvaluationRun, dataset: EvaluationDataset) -> None: ...


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
            raise EvaluationRunError("Cloud Run did not accept the evaluation job dispatch")


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
