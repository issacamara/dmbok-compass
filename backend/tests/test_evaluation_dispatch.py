from __future__ import annotations

from dataclasses import dataclass, field

from google.api_core.exceptions import AlreadyExists

from app.contracts import EvaluationDataset, EvaluationItem, EvaluationRun
from app.evaluation.dispatch import CloudRunEvaluationJobDispatcher, EvaluationDispatchClaimStore


@dataclass
class Document:
    records: dict[str, dict[str, str]]
    path: str

    def create(self, payload: dict[str, str]) -> None:
        if self.path in self.records:
            raise AlreadyExists("already claimed")
        self.records[self.path] = payload

    def delete(self) -> None:
        self.records.pop(self.path, None)


@dataclass
class Collection:
    records: dict[str, dict[str, str]]
    path: str

    def document(self, run_id: str) -> Document:
        return Document(self.records, f"{self.path}/{run_id}")


@dataclass
class Firestore:
    records: dict[str, dict[str, str]] = field(default_factory=dict)

    def collection(self, collection_id: str) -> Collection:
        return Collection(self.records, collection_id)


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
