"""Publish and launch immutable exploratory runs for a candidate revision."""

from __future__ import annotations

from hashlib import sha256
import json

from app.contracts import EvaluationCandidateSnapshot, EvaluationDataset
from app.contracts.v29 import CandidateDatasetBinding, DatasetHead, EvaluationRunSnapshot
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.job import EvaluationJobService, EvaluationRunError, candidate_rows_dataset
from model.candidate_generation import PROMPT_VERSION


class CandidateExplorationPublisher:
    """Bind a committed row-v2 revision to one durable, exploratory run."""

    def __init__(self, store: FirestoreEvaluationGenerationStore, jobs: EvaluationJobService) -> None:
        self.store = store
        self.jobs = jobs

    def launch(
        self,
        *,
        attempt_id: str,
        head: DatasetHead,
        configuration_version_id: str,
        model_version_id: str,
        item_ids: list[str] | None = None,
    ) -> EvaluationRunSnapshot:
        if self.store.corpus_head(head.corpus_version_id) != head:
            raise EvaluationRunError("candidate dataset revision is no longer current")
        rows = self.store.get_revision(head)
        if not rows:
            raise EvaluationRunError("candidate dataset revision is empty")
        dataset_id = _dataset_id(head)
        dataset = candidate_rows_dataset(dataset_id, rows)
        self.store.stage(dataset)

        candidate = self.jobs.snapshots.resolve(
            corpus_version_id=head.corpus_version_id,
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
        )
        selected_ids = tuple(item.item_id for item in dataset.items) if item_ids is None else tuple(item_ids)
        binding = CandidateDatasetBinding(
            attempt_id=attempt_id,
            corpus_version_id=head.corpus_version_id,
            dataset_revision=head.revision,
        )
        run_id = _candidate_run_id(
            binding, configuration_version_id, model_version_id, candidate, selected_ids,
            selection="full" if item_ids is None else "subset",
        )
        snapshot = self.store.create_run_snapshot(
            run_id=run_id,
            binding=binding,
            configuration_version_id=configuration_version_id,
            configuration=candidate,
            item_ids=None if item_ids is None else selected_ids,
        )
        self.jobs.launch_immutable_candidate(
            dataset=dataset,
            corpus_version_id=head.corpus_version_id,
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
            snapshot=candidate,
            run_id=run_id,
            item_ids=list(selected_ids),
        )
        return snapshot


def candidate_generator_fingerprint(route: str, model: str, configuration_version: str) -> str:
    value = "\0".join((PROMPT_VERSION, route, model, configuration_version))
    return sha256(value.encode("utf-8")).hexdigest()


def _dataset_id(head: DatasetHead) -> str:
    identity = f"{head.corpus_version_id}\0{head.revision}\0{head.dataset_sha256}"
    return "candidate-" + sha256(identity.encode("utf-8")).hexdigest()


def _candidate_run_id(
    binding: CandidateDatasetBinding,
    configuration_version_id: str,
    model_version_id: str,
    snapshot: EvaluationCandidateSnapshot,
    selected_ids: tuple[str, ...],
    *,
    selection: str,
) -> str:
    material = {
        "binding": binding.model_dump(),
        "configuration_version_id": configuration_version_id,
        "model_version_id": model_version_id,
        "snapshot": snapshot.model_dump(mode="json"),
        "selection": selection,
        "selected_item_ids": selected_ids,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return "eval-" + sha256(encoded.encode("utf-8")).hexdigest()[:32]
