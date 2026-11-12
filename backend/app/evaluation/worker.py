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
from app.corpus.versions import legacy_serving_binding
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.assessments import score_assessment
from app.evaluation.job import (
    EvaluationCandidateSnapshotResolver,
    ResolvedEvaluationCandidate,
    ResolvedServingEvaluation,
    EvaluationOutcome,
    EvaluationRunError,
    candidate_rows_dataset,
    EvaluationCorpusReadinessError,
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

_CANDIDATE_DATASET_TARGET = 200


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
        try:
            configuration = RuntimeConfiguration.from_environment()
        except ValueError as exc:
            raise EvaluationRunError(str(exc)) from exc
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
        if corpus is None:
            raise EvaluationRunError("uploaded corpus is unavailable")
        if corpus.validation_evidence is None or corpus.serving_binding is None:
            raise EvaluationCorpusReadinessError(
                "evaluation corpus provenance evidence is unavailable"
            )
        binding = corpus.serving_binding or legacy_serving_binding(
            corpus, vector_collection=factory.settings.vector_collection
        )
        request_timeout_ms = configuration.request_timeout_ms
        generation_timeout_ms = configuration.generation_timeout_ms
        try:
            return factory.candidate_snapshot(
                binding,
                request_timeout_ms=request_timeout_ms,
                generation_timeout_ms=generation_timeout_ms,
            )
        except ValueError as exc:
            raise EvaluationRunError(str(exc)) from exc

    def resolve_candidate(self) -> ResolvedEvaluationCandidate:
        try:
            configuration = RuntimeConfiguration.from_environment()
        except ValueError as exc:
            raise EvaluationRunError(str(exc)) from exc
        runtime = configuration.runtime
        corpus_store = FirestoreCorpusVersionStore(
            self.client, vector_collection=configuration.vector_collection
        )
        corpus = corpus_store.active()
        if corpus is None:
            raise EvaluationRunError("active corpus is unavailable")
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

    def resolve_serving(self) -> ResolvedServingEvaluation:
        generations = FirestoreEvaluationGenerationStore(self.client)
        binding, dataset_id, snapshot, head, rows, _, configuration_version_id, model_version_id = (
            generations.resolve_serving_revision()
        )
        dataset = candidate_rows_dataset(dataset_id, rows)
        stored_dataset = generations.get(dataset_id)
        if stored_dataset is None or stored_dataset.model_dump(mode="json") != dataset.model_dump(mode="json"):
            raise EvaluationRunError("serving dataset does not match the approved revision")
        return ResolvedServingEvaluation(
            binding=binding,
            dataset=dataset,
            configuration_version_id=configuration_version_id,
            model_version_id=model_version_id,
            snapshot=snapshot,
            head=head,
            rows=rows,
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
    current_corpus_available: Callable[[], bool] | None = None,
) -> None:
    """Reload the accepted dataset and persist one aggregate terminal outcome."""

    run = runs.get(run_id)
    if run is None:
        raise EvaluationRunError("unknown evaluation run")
    if current_corpus_available is not None and not current_corpus_available():
        runs.update(run.model_copy(update={"status": "failed", "error": "evaluation corpus unavailable"}))
        return
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
    configuration = RuntimeConfiguration.from_environment()
    project = configuration.runtime.project
    client = firestore.Client(project=project)
    task = os.environ.get("EVALUATION_TASK", "run").strip()
    if task == "prepare_candidates":
        try:
            _prepare_candidate_evaluation(client, project, configuration)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        return
    if task != "run":
        raise EvaluationRunError("unknown evaluation task")
    run_id = _required("EVALUATION_RUN_ID")
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
                current_corpus_available=lambda: _current_corpus_available(client, configuration.vector_collection),
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
            current_corpus_available=lambda: _current_corpus_available(client, configuration.vector_collection),
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


def _prepare_candidate_evaluation(client: firestore.Client, project: str, runtime_configuration) -> None:
    """Generate once for a validated corpus, publish its head, then launch its run."""
    from uuid import uuid4

    from app.evaluation.automatic import CandidateExplorationPublisher, candidate_generator_fingerprint
    from app.evaluation.candidate_generation import generate_candidate_revision
    from app.evaluation.dispatch import CloudRunCandidateGenerationDispatcher, EvaluationDispatchClaimStore
    from app.corpus.staging import FirestoreCorpusStagingAttemptStore
    from app.telemetry import LoggingTelemetryAdapter, stage_event
    from app.provider_allowance import ProviderAccountCapabilities
    from app.provider_allowance_firestore import FirestoreProviderAllowanceStore
    from app.settings import EvaluationStageConfiguration
    from model.candidate_generation import EvaluationCandidateAdapter

    attempt_id = _required("CORPUS_STAGING_ATTEMPT_ID")
    corpus_version_id = _required("EVALUATION_CORPUS_VERSION_ID")
    dispatch_owner_id = _required("EVALUATION_DISPATCH_OWNER_ID")
    staging = FirestoreCorpusStagingAttemptStore(client)
    telemetry = LoggingTelemetryAdapter()
    attempt = staging.get(attempt_id=attempt_id)
    _validate_candidate_staging_attempt(attempt, attempt_id=attempt_id, corpus_version_id=corpus_version_id)
    staging.update_candidate_generation_status(attempt_id=attempt_id, state="queued")
    stage = EvaluationStageConfiguration.from_environment("GENERATOR")
    fingerprint = candidate_generator_fingerprint(
        stage.route, stage.model, stage.configuration_version
    )
    if _required("EVALUATION_GENERATOR_FINGERPRINT") != fingerprint:
        raise EvaluationRunError("candidate generator configuration changed before preparation")
    operation_key = CloudRunCandidateGenerationDispatcher.operation_key(corpus_version_id, fingerprint)
    operation_claim_id = CloudRunCandidateGenerationDispatcher.claim_id(operation_key)
    store = FirestoreEvaluationGenerationStore(client)
    committed_head = store.operation_head(operation_key, corpus_version_id=corpus_version_id)
    if committed_head is not None:
        CandidateExplorationPublisher(store, _configured_evaluation_service()).launch(
            attempt_id=attempt_id,
            head=committed_head,
            configuration_version_id=stage.configuration_version,
            model_version_id=stage.model,
        )
        committed_rows = store.get_revision(committed_head)
        completed_count = len(committed_rows)
        target_count = attempt.get("candidate_generation_target")
        if type(target_count) is not int or target_count < completed_count:
            target_count = _CANDIDATE_DATASET_TARGET
        rejected_count = attempt.get("candidate_generation_rejected_count")
        if type(rejected_count) is not int or rejected_count < 0:
            rejected_count = None
        usage_tokens = attempt.get("candidate_generation_usage_tokens")
        if type(usage_tokens) is not int or usage_tokens < 0:
            usage_tokens = None
        staging.update_candidate_generation_status(
            attempt_id=attempt_id,
            state="published",
            completed_count=completed_count,
            target_count=target_count,
            rejected_count=rejected_count,
            usage_tokens=usage_tokens,
        )
        telemetry.emit(stage_event(
            "generation", "succeeded",
            completed_count=completed_count,
            total_count=target_count,
            usage_kind="tokens" if usage_tokens is not None else None,
            usage_units=usage_tokens,
        ))
        return
    owner_id = uuid4().hex
    if not store.claim_operation(
        operation_key, corpus_version_id=corpus_version_id, owner_id=owner_id
    ):
        return
    staging.update_candidate_generation_status(attempt_id=attempt_id, state="running")
    telemetry.emit(stage_event("generation", "started"))
    claims = EvaluationDispatchClaimStore(client)
    allowances = FirestoreProviderAllowanceStore(client)
    provider_id = "google-gemini" if stage.route == "vertex" else stage.route
    capabilities = ProviderAccountCapabilities(
        provider_id=provider_id,
        account_id=stage.allowance_policy,
        kind="generation",
        max_units_per_operation=stage.allowance_max_units,
        max_retries=1,
    )
    try:
        reservation = allowances.stage_reservation(
            capabilities, operation_key, stage.allowance_max_units, retries=0
        )
        if reservation.status != "staged":
            store.release_operation(operation_key, owner_id=owner_id, status="awaiting_budget")
            claims.release(operation_claim_id, owner_id=dispatch_owner_id)
            staging.update_candidate_generation_status(attempt_id=attempt_id, state="awaiting_budget")
            telemetry.emit(stage_event("generation", "awaiting_budget"))
            print(json.dumps({"event": "candidate_generation_paused", "status": "awaiting_budget"}))
            return
        records = store.get_parse_derived_records(corpus_version_id=corpus_version_id)
        if records is None:
            raise EvaluationRunError("candidate corpus is not complete and validated")
        parents, children = records
        adapter = EvaluationCandidateAdapter(
            route=stage.route,
            model=stage.model,
            timeout_ms=runtime_configuration.generation_timeout_ms,
            max_output_tokens=runtime_configuration.output_token_cap,
            api_key=os.environ.get("OPENROUTER_API_KEY") if stage.route == "openrouter" else None,
            project=project,
            location=runtime_configuration.runtime.location,
        )
        def record_progress(completed: int, target: int, usage_tokens: int | None, rejected: int) -> None:
            staging.update_candidate_generation_status(
                attempt_id=attempt_id, state="running", completed_count=completed,
                target_count=target, rejected_count=rejected, usage_tokens=usage_tokens,
            )
            telemetry.emit(stage_event(
                "generation", "started", completed_count=completed, total_count=target,
                usage_kind="tokens" if usage_tokens is not None else None,
                usage_units=usage_tokens,
            ))

        report = asyncio.run(generate_candidate_revision(
            corpus_version_id=corpus_version_id,
            children=children,
            parents=parents,
            corpus_store=FirestoreCorpusVersionStore(
                client, vector_collection=runtime_configuration.vector_collection
            ),
            adapter=adapter,
            generator_route=stage.route,
            generator_model=stage.model,
            generator_configuration_version=stage.configuration_version,
            operation_key=operation_key,
            operation_owner_id=owner_id,
            progress_callback=record_progress,
            allowance=reservation,
            allowance_store=allowances,
            allowance_capabilities=capabilities,
            store=store,
            target_count=_CANDIDATE_DATASET_TARGET,
        ))
        if report.status == "awaiting_budget":
            store.release_operation(operation_key, owner_id=owner_id, status="awaiting_budget")
            claims.release(operation_claim_id, owner_id=dispatch_owner_id)
            staging.update_candidate_generation_status(attempt_id=attempt_id, state="awaiting_budget")
            telemetry.emit(stage_event("generation", "awaiting_budget"))
            print(json.dumps({"event": "candidate_generation_paused", "status": "awaiting_budget"}))
            return
        if report.status != "published" or report.dataset_head is None:
            raise EvaluationRunError("candidate dataset generation failed")
        CandidateExplorationPublisher(store, _configured_evaluation_service()).launch(
            attempt_id=attempt_id,
            head=report.dataset_head,
            configuration_version_id=stage.configuration_version,
            model_version_id=stage.model,
        )
        staging.update_candidate_generation_status(
            attempt_id=attempt_id, state="published", completed_count=report.generated_count,
            target_count=report.target_count, rejected_count=report.rejected_count,
            usage_tokens=report.usage_tokens,
        )
        telemetry.emit(stage_event(
            "generation", "succeeded", completed_count=report.generated_count,
            total_count=report.target_count,
            usage_kind="tokens" if report.usage_tokens is not None else None,
            usage_units=report.usage_tokens,
        ))
        print(json.dumps({
            "event": "candidate_exploration_dispatched",
            "status": "published",
            "generated_count": report.generated_count,
        }))
    except Exception as exc:
        store.release_operation(operation_key, owner_id=owner_id, status="failed")
        claims.release(operation_claim_id, owner_id=dispatch_owner_id)
        staging.update_candidate_generation_status(attempt_id=attempt_id, state="failed")
        telemetry.emit(stage_event("generation", "failed", error=exc))
        raise RuntimeError("candidate dataset generation failed") from None


def _configured_evaluation_service():
    from app.main import get_evaluation_service

    return get_evaluation_service()


def _validate_candidate_staging_attempt(
    attempt: dict[str, object] | None, *, attempt_id: str, corpus_version_id: str
) -> None:
    """Require the durable staged attempt to bind the independently supplied corpus ID."""
    if (
        attempt is None
        or attempt.get("attempt_id") != attempt_id
        or attempt.get("state") != "staged"
        or attempt.get("staged_corpus_version_id") != corpus_version_id
    ):
        raise EvaluationRunError("candidate staging attempt does not match its validated corpus")




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
    binding = _effective_binding(
        corpus, vector_collection=snapshot.corpus_binding.vector_collection
    )
    if binding != snapshot.corpus_binding:
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
    if corpus is None or corpus.validation_evidence is None or corpus.serving_binding is None:
        raise EvaluationCorpusReadinessError(
            "evaluation corpus provenance evidence is unavailable"
        )
    binding = _effective_binding(
        corpus, vector_collection=snapshot.corpus_binding.vector_collection
    )
    if binding != snapshot.corpus_binding:
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


def _effective_binding(corpus, *, vector_collection: str):
    return corpus.serving_binding or legacy_serving_binding(
        corpus, vector_collection=vector_collection
    )


def _current_corpus_available(client: firestore.Client, vector_collection: str) -> bool:
    """Return only availability; never persist or log the corpus identity."""

    try:
        corpus = FirestoreCorpusVersionStore(client, vector_collection=vector_collection).active()
    except Exception:
        return False
    return corpus is not None


def _unavailable() -> EvaluationOutcome:
    raise EvaluationRunError("evaluation dataset is unavailable")


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    main()
