from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import AlreadyExists

from app.contracts import EvaluationDataset, EvaluationItem, EvaluationRun
from app.evaluation.dispatch import (
    CloudRunCandidateGenerationDispatcher,
    CloudRunEvaluationJobDispatcher,
    CloudRunJobExecutor,
    EvaluationDispatchClaimStore,
    EvaluationDispatchError,
)


@dataclass
class Document:
    records: dict[str, dict[str, object]]
    path: str

    def create(self, payload: dict[str, str]) -> None:
        if self.path in self.records:
            raise AlreadyExists("already claimed")
        self.records[self.path] = payload

    def delete(self) -> None:
        self.records.pop(self.path, None)

    def get(self, *, transaction=None):
        records = transaction.data if transaction is not None else self.records
        return Snapshot(records.get(self.path))

    def update(self, payload: dict[str, object]) -> None:
        self.records[self.path] = {**self.records[self.path], **payload}


@dataclass
class Snapshot:
    data: dict[str, object] | None

    @property
    def exists(self):
        return self.data is not None

    def to_dict(self):
        return deepcopy(self.data) if self.data is not None else {}


@dataclass
class Collection:
    records: dict[str, dict[str, object]]
    path: str

    def document(self, run_id: str) -> Document:
        return Document(self.records, f"{self.path}/{run_id}")


@dataclass
class Firestore:
    records: dict[str, dict[str, object]] = field(default_factory=dict)

    def collection(self, collection_id: str) -> Collection:
        return Collection(self.records, collection_id)

    def transaction(self):
        return Transaction(self, deepcopy(self.records))


@dataclass
class Transaction:
    client: Firestore
    data: dict[str, dict[str, object]]

    def set(self, reference: Document, payload: dict[str, object]) -> None:
        self.data[reference.path] = deepcopy(payload)

    def update(self, reference: Document, payload: dict[str, object]) -> None:
        self.data[reference.path] = {**self.data[reference.path], **payload}

    def delete(self, reference: Document) -> None:
        self.data.pop(reference.path, None)


def _run() -> EvaluationRun:
    return EvaluationRun(
        run_id="eval-1", dataset_version_id="generation-1", corpus_version_id="corpus-1", status="queued",
        selected_item_ids=["item-1"], configuration_version_id="config-1", model_version_id="model-1",
    )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset_version_id="generation-1",
        items=[EvaluationItem(item_id="item-1", question="Question", dataset_version_id="generation-1")],
    )


def test_dispatch_claim_survives_dispatcher_restart_and_sends_only_the_run_id() -> None:
    firestore = Firestore()
    invoked: list[str] = []
    first = CloudRunEvaluationJobDispatcher(EvaluationDispatchClaimStore(firestore), invoked.append)
    restarted = CloudRunEvaluationJobDispatcher(EvaluationDispatchClaimStore(firestore), invoked.append)

    first.dispatch(_run(), _dataset())
    restarted.dispatch(_run(), _dataset())

    assert invoked == ["eval-1"]
    assert firestore.records["evaluation_run_dispatches/eval-1"]["run_id"] == "eval-1"


def test_failed_dispatch_releases_claim_for_safe_retry() -> None:
    firestore = Firestore()
    attempts = 0

    def fail_once(run_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("unavailable")

    dispatcher = CloudRunEvaluationJobDispatcher(EvaluationDispatchClaimStore(firestore), fail_once)
    try:
        dispatcher.dispatch(_run(), _dataset())
    except RuntimeError:
        pass
    dispatcher.dispatch(_run(), _dataset())

    assert attempts == 2


def test_candidate_preparation_dispatch_releases_failed_submission_for_retry() -> None:
    class Claims:
        def __init__(self):
            self.active = {}
            self.completed = set()

        def claim_leased(self, claim_id):
            if claim_id in self.completed or claim_id in self.active:
                return None
            owner_id = f"owner-{len(self.active) + len(self.completed) + 1}"
            self.active[claim_id] = owner_id
            return owner_id

        def release(self, claim_id, *, owner_id=None):
            if self.active.get(claim_id) == owner_id:
                self.active.pop(claim_id, None)

        def complete_claim(self, claim_id, *, owner_id):
            assert self.active.get(claim_id) == owner_id
            self.completed.add(claim_id)
            self.active.pop(claim_id, None)

    class Executor:
        def __init__(self):
            self.calls = 0
            self.environments = []

        def dispatch_overrides(self, environment):
            self.calls += 1
            self.environments.append(environment)
            if self.calls == 1:
                raise RuntimeError("temporary dispatch failure")

    claims = Claims()
    executor = Executor()
    dispatcher = CloudRunCandidateGenerationDispatcher(claims, executor)
    arguments = {
        "attempt_id": "attempt-1",
        "corpus_version_id": "corpus-1",
        "generator_fingerprint": "fingerprint-1",
    }

    import pytest
    with pytest.raises(RuntimeError, match="temporary dispatch failure"):
        dispatcher.dispatch(**arguments)
    assert dispatcher.dispatch(**arguments) is True
    assert dispatcher.dispatch(**arguments) is False
    assert executor.calls == 2
    assert executor.environments[1]["EVALUATION_DISPATCH_OWNER_ID"]


def test_dispatch_claim_lease_can_be_recovered_after_expiry(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def inline_transaction(function):
        def execute(transaction):
            result = function(transaction)
            transaction.client.records.clear()
            transaction.client.records.update(transaction.data)
            return result

        return execute

    monkeypatch.setattr("google.cloud.firestore.transactional", inline_transaction)
    client = Firestore()
    claims = EvaluationDispatchClaimStore(client)
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    stale_owner = claims.claim_leased("candidate-task", lease_seconds=30, now=now)
    assert stale_owner is not None
    assert claims.claim_leased("candidate-task", lease_seconds=30, now=now + timedelta(seconds=20)) is None
    current_owner = claims.claim_leased("candidate-task", lease_seconds=30, now=now + timedelta(seconds=31))
    assert current_owner is not None and current_owner != stale_owner

    # An expired sender must not erase or complete the newer sender's lease.
    claims.release("candidate-task", owner_id=stale_owner)
    claims.complete_claim("candidate-task", owner_id=stale_owner)
    record = client.records["evaluation_run_dispatches/candidate-task"]
    assert record["owner_id"] == current_owner
    assert record["state"] == "submitting"

    claims.complete_claim("candidate-task", owner_id=current_owner)
    claims.release("candidate-task", owner_id=stale_owner)
    assert client.records["evaluation_run_dispatches/candidate-task"]["state"] == "submitted"
    assert claims.claim_leased("candidate-task", lease_seconds=30, now=now + timedelta(minutes=1)) is None
    claims.release("candidate-task", owner_id=current_owner)
    assert "evaluation_run_dispatches/candidate-task" not in client.records
    assert claims.claim_leased("candidate-task", lease_seconds=30, now=now + timedelta(minutes=1)) is not None


def test_cloud_run_executor_sends_run_override_and_returns_bounded_provider_diagnostic(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class Response:
        ok = False
        status_code = 403

        def json(self) -> dict[str, object]:
            return {"error": {"status": "PERMISSION_DENIED", "message": "secret evaluator content"}}

    class Session:
        def __init__(self, credentials: object) -> None:
            del credentials

        def post(self, url: str, *, json: object, timeout: int) -> Response:
            captured.update(url=url, json=json, timeout=timeout)
            return Response()

    monkeypatch.setattr("google.auth.default", lambda scopes: (object(), object()))
    monkeypatch.setattr("google.auth.transport.requests.AuthorizedSession", Session)

    import pytest

    with pytest.raises(EvaluationDispatchError) as raised:
        CloudRunJobExecutor(project="project", location="region", job_name="evaluation")("run-1")

    assert captured["url"] == "https://run.googleapis.com/v2/projects/project/locations/region/jobs/evaluation:run"
    assert captured["json"] == {
        "overrides": {"containerOverrides": [{"env": [{"name": "EVALUATION_RUN_ID", "value": "run-1"}]}]}
    }
    assert captured["timeout"] == 15
    assert raised.value.status_code == 403
    assert raised.value.provider_code == "PERMISSION_DENIED"
    assert raised.value.provider_message == "Cloud Run rejected the caller permissions."
    assert "secret evaluator content" not in str(raised.value)
