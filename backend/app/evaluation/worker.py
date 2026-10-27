"""Cloud Run worker for one immutable, durable evaluation run."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Callable

from google.cloud import firestore

from app.answering import (
    GroundedAnswerPolicy,
    RetrievedAnswerService,
)
from app.contracts import EvaluationDataset, EvaluationRun, QuotaStatus, RunBoundSupportEvidence
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.assessments import score_assessment
from app.evaluation.job import (
    EvaluationCandidateSnapshotResolver,
    ResolvedEvaluationCandidate,
    EvaluationOutcome,
    EvaluationRunError,
    EvaluationRunStore,
    EvaluationStageStore,
    FirestoreEvaluationStageStore,
    EvaluationInterrupted,
    EvaluationWorker,
)
from app.evaluation.job import FirestoreEvaluationRunStore
from app.evaluation.support import (
    FirestoreRunPinnedChunkSource,
    FirestoreSupportEvidenceStore,
    prepare_support,
)
from app.rag import LangChainRagFactory, LangChainRagSettings
from app.settings import RuntimeConfiguration


class FirestoreEvaluationCandidateSnapshotResolver(EvaluationCandidateSnapshotResolver):
    """Resolve an evaluation candidate once, before durable dispatch."""

    def __init__(self, client: firestore.Client) -> None:
        self.client = client

    def resolve(
        self,
        *,
        corpus_version_id: str,
        configuration_version_id: str,
        model_version_id: str,
    ):
        configuration = RuntimeConfiguration.from_environment()
        runtime = configuration.runtime
        factory = LangChainRagFactory(self.client, LangChainRagSettings(
            project=runtime.project,
            location=runtime.location,
            vector_collection=configuration.vector_collection,
            answer_provider=runtime.provider,
            primary_model=runtime.primary_model,
            fallback_model=runtime.fallback_model,
            json_schema_supported=configuration.json_schema_supported,
            output_token_cap=configuration.output_token_cap,
        ))
        corpus = FirestoreCorpusVersionStore(
            self.client, vector_collection=factory.settings.vector_collection
        ).get(corpus_version_id)
        if corpus is None or corpus.serving_binding is None:
            raise EvaluationRunError("candidate corpus serving binding is unavailable")
        request_timeout_ms = configuration.request_timeout_ms
        generation_timeout_ms = configuration.generation_timeout_ms
        try:
            return factory.candidate_snapshot(
                corpus.serving_binding,
                request_timeout_ms=request_timeout_ms,
                generation_timeout_ms=generation_timeout_ms,
            )
        except ValueError as exc:
            raise EvaluationRunError(str(exc)) from exc

    def resolve_candidate(self) -> ResolvedEvaluationCandidate:
        configuration = RuntimeConfiguration.from_environment()
        runtime = configuration.runtime
        corpus_store = FirestoreCorpusVersionStore(
            self.client, vector_collection=configuration.vector_collection
        )
        corpus = corpus_store.active()
        if corpus is None or corpus.serving_binding is None:
            raise EvaluationRunError("active corpus serving binding is unavailable")
        configuration_version_id = os.environ.get("CONFIGURATION_VERSION_ID", "").strip()
        if not configuration_version_id:
            raise EvaluationRunError("runtime configuration version is unavailable")
        model_version_id = os.environ.get("MODEL_VERSION_ID", "").strip() or runtime.primary_model
        snapshot = self.resolve(
            corpus_version_id=corpus.version_id,
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
        )
        return ResolvedEvaluationCandidate(
            corpus_version_id=corpus.version_id,
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
            snapshot=snapshot,
        )


def execute_run(
    run_id: str,
    *,
    datasets: FirestoreEvaluationGenerationStore,
    runs: EvaluationRunStore,
    evaluate: Callable[[str], EvaluationOutcome],
    termination_requested: Callable[[], bool] | None = None,
    stages: EvaluationStageStore | None = None,
    support_preparer: Callable[[object, EvaluationDataset], object] | None = None,
    support_evidence: Callable[..., RunBoundSupportEvidence | None] | None = None,
) -> None:
    """Reload the run-pinned generation and persist its terminal outcome."""

    run = runs.get(run_id)
    if run is None:
        raise EvaluationRunError("unknown evaluation run")
    dataset = datasets.get(run.dataset_version_id)
    if dataset is None:
        runs.update(run.model_copy(update={"status": "failed", "error": "evaluation dataset is unavailable"}))
        return
    EvaluationWorker(runs, stages=stages).execute(
        run_id,
        dataset,
        evaluate,
        termination_requested=termination_requested,
        support_preparer=support_preparer,
        support_evidence=support_evidence,
    )


class EvaluationAnswerEvaluator:
    """Produce content-free evaluation outcomes from the pinned corpus answer path."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        answerer: RetrievedAnswerService,
        *,
        run: EvaluationRun | None = None,
        support_evidence: Callable[..., RunBoundSupportEvidence | None] | None = None,
        scorer_version: str = "evaluation-gates-v1",
    ) -> None:
        self._items = {item.item_id: item for item in dataset.items}
        self._annotations = {annotation.item_id: annotation for annotation in dataset.gold_annotations}
        self._answerer = answerer
        self._run = run
        self._support_evidence = support_evidence
        self._scorer_version = scorer_version

    def __call__(self, item_id: str) -> EvaluationOutcome:
        item = self._items[item_id]
        response = asyncio.run(self._answerer.answer(item.question, _evaluation_quota()))
        annotation = self._annotations.get(item_id)
        retrieved = {passage.chunk_id for passage in response.trace.retrieved_passages}
        cited = {citation.citation_id for citation in response.citations}
        score = score_assessment(
            annotation,
            outcome=response.outcome,
            answer_text=response.answer_text,
            retrieved_chunk_ids=retrieved,
            cited_chunk_ids=cited,
            scorer_version=self._scorer_version,
            operational_error=response.error is not None,
            run=self._run,
            support_evidence=(
                self._support_evidence(run_id=self._run.run_id, item_id=item_id)
                if self._run is not None and self._support_evidence is not None
                else None
            ),
        )
        return EvaluationOutcome(
            retrieval_success=score.retrieval_success,
            grounded=score.grounded,
            citation_correct=score.citation_correct,
            answer_quality=score.answer_quality,
            refusal_correct=score.refusal_correct,
            response_time_ms=float(response.trace.timings_ms.get("total_ms", 0)),
            applicable_metrics=score.applicable_metrics,
        )


def main() -> None:
    termination_requested = Event()
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: termination_requested.set())
    project = RuntimeConfiguration.from_environment().runtime.project
    run_id = _required("EVALUATION_RUN_ID")
    client = firestore.Client(project=project)
    datasets = FirestoreEvaluationGenerationStore(client)
    runs = FirestoreEvaluationRunStore(client)
    stages = FirestoreEvaluationStageStore(client)
    run = runs.get(run_id)
    if run is None:
        raise EvaluationRunError("unknown evaluation run")
    try:
        dataset = datasets.get(run.dataset_version_id)
        if dataset is None:
            execute_run(
                run_id,
                datasets=datasets,
                runs=runs,
                evaluate=lambda _: _unavailable(),
                termination_requested=termination_requested.is_set,
                stages=stages,
            )
            return
        support_store = FirestoreSupportEvidenceStore(client)
        execute_run(
            run_id,
            datasets=datasets,
            runs=runs,
            evaluate=EvaluationAnswerEvaluator(
                dataset,
                _pinned_answerer(project, run),
                run=run,
                support_evidence=support_store.get_evidence,
                scorer_version=run.candidate_snapshot.scorer_version,
            ),
            termination_requested=termination_requested.is_set,
            stages=stages,
            support_preparer=lambda pinned_run, pinned_dataset: _prepare_run_support(client, pinned_run, pinned_dataset),
            support_evidence=support_store.get_evidence,
        )
    except EvaluationInterrupted:
        print(json.dumps({"event": "dmbok_evaluation_interrupted", "exit_code": 1}))
        raise SystemExit(1) from None
    except Exception:
        # Configuration and provider setup failures must be durable, but may
        # still exit non-zero so Cloud Run applies its bounded retry policy.
        runs.update(run.model_copy(update={"status": "failed", "error": "evaluation worker setup failed"}))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _pinned_answerer(project: str, run) -> RetrievedAnswerService:
    snapshot = run.candidate_snapshot
    if snapshot is None:
        raise EvaluationRunError("evaluation run has no immutable candidate snapshot")
    if snapshot.project != project:
        raise EvaluationRunError("evaluation worker project does not match the candidate snapshot")
    client = firestore.Client(project=project)
    from app.rag import LangChainRagSettings

    factory = LangChainRagFactory(
        client,
        LangChainRagSettings(
            project=snapshot.project,
            location=snapshot.location,
            vector_collection=snapshot.corpus_binding.vector_collection,
            answer_provider=snapshot.answer_provider,
        ),
    )
    corpus = FirestoreCorpusVersionStore(
        client, vector_collection=snapshot.corpus_binding.vector_collection
    ).get(run.corpus_version_id)
    if corpus is None:
        raise EvaluationRunError("run references an unavailable corpus version")
    if corpus.serving_binding != snapshot.corpus_binding:
        raise EvaluationRunError("candidate corpus binding no longer matches the run snapshot")
    adapter = factory.generation_adapter_for_snapshot(snapshot)

    class _PinnedCorpus:
        def active(self):
            return corpus

    return RetrievedAnswerService(
        factory.retriever(),
        _PinnedCorpus(),
        GroundedAnswerPolicy(
            adapter,
            model_name=snapshot.primary_model,
            timeout_ms=snapshot.generation_timeout_ms,
        ),
        request_timeout_ms=snapshot.request_timeout_ms,
        retriever_for_binding=lambda pinned: factory.retriever_for_binding(
            snapshot.corpus_binding
        ),
    )


def _prepare_run_support(client: firestore.Client, run, dataset: EvaluationDataset) -> object:
    """Prepare evidence against the same immutable corpus binding as scoring."""

    snapshot = run.candidate_snapshot
    if snapshot is None:
        raise EvaluationRunError("evaluation run has no immutable candidate snapshot")
    corpus = FirestoreCorpusVersionStore(
        client, vector_collection=snapshot.corpus_binding.vector_collection
    ).get(run.corpus_version_id)
    if corpus is None or corpus.validation_evidence is None:
        raise EvaluationRunError("run corpus provenance evidence is unavailable")
    if corpus.serving_binding != snapshot.corpus_binding:
        raise EvaluationRunError("candidate corpus binding no longer matches the run snapshot")
    source = FirestoreRunPinnedChunkSource(
        client,
        vector_collection=snapshot.corpus_binding.vector_collection,
        provenance_by_corpus={run.corpus_version_id: corpus.validation_evidence.provenance_sha256},
        ingestion_fingerprint=snapshot.corpus_binding.ingestion_fingerprint,
        configuration_fingerprint=snapshot.corpus_binding.configuration_fingerprint,
    )
    return prepare_support(run, dataset, source, FirestoreSupportEvidenceStore(client))


def _evaluation_quota() -> QuotaStatus:
    return QuotaStatus(
        user_used=0,
        user_limit=0,
        global_used=0,
        global_limit=0,
        resets_at=datetime.now(timezone.utc) + timedelta(days=1),
    )


def _unavailable() -> EvaluationOutcome:
    raise EvaluationRunError("evaluation dataset is unavailable")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
