"""Cloud Run worker for one immutable, durable evaluation run."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Callable

from google.cloud import firestore

from app.answering import (
    DEFAULT_GENERATION_TIMEOUT_MS,
    DEFAULT_REQUEST_TIMEOUT_MS,
    GroundedAnswerPolicy,
    RetrievedAnswerService,
    timeout_settings,
)
from app.contracts import EvaluationDataset, QuotaStatus
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.corpus.versions import legacy_serving_binding
from app.evaluation.generations import FirestoreEvaluationGenerationStore
from app.evaluation.job import EvaluationOutcome, EvaluationRunError, EvaluationRunStore, EvaluationWorker
from app.evaluation.job import FirestoreEvaluationRunStore
from app.rag import LangChainRagFactory


def execute_run(
    run_id: str,
    *,
    datasets: FirestoreEvaluationGenerationStore,
    runs: EvaluationRunStore,
    evaluate: Callable[[str], EvaluationOutcome],
) -> None:
    """Reload the run-pinned generation and persist its terminal outcome."""

    run = runs.get(run_id)
    if run is None:
        raise EvaluationRunError("unknown evaluation run")
    dataset = datasets.get(run.dataset_version_id)
    if dataset is None:
        runs.update(run.model_copy(update={"status": "failed", "error": "evaluation dataset is unavailable"}))
        return
    EvaluationWorker(runs).execute(run_id, dataset, evaluate)


class EvaluationAnswerEvaluator:
    """Produce content-free evaluation outcomes from the pinned corpus answer path."""

    def __init__(self, dataset: EvaluationDataset, answerer: RetrievedAnswerService) -> None:
        self._items = {item.item_id: item for item in dataset.items}
        self._annotations = {annotation.item_id: annotation for annotation in dataset.gold_annotations}
        self._answerer = answerer

    def __call__(self, item_id: str) -> EvaluationOutcome:
        item = self._items[item_id]
        response = asyncio.run(self._answerer.answer(item.question, _evaluation_quota()))
        annotation = self._annotations.get(item_id)
        retrieved = {passage.chunk_id for passage in response.trace.retrieved_passages}
        cited = {citation.citation_id for citation in response.citations}
        expected = set(annotation.relevant_chunk_ids) if annotation is not None else set()
        is_answer = response.outcome in {"answer", "qualified"}
        # Candidate-only datasets are explicitly exploratory. Gold annotations
        # give retrieval and citation scoring a durable human-reviewed oracle.
        retrieval_success = bool(retrieved & expected) if expected else bool(retrieved)
        citation_correct = cited <= expected and bool(cited) if expected else bool(cited)
        grounded = is_answer and bool(cited)
        answer_quality = is_answer and bool(response.answer_text)
        refusal_correct = (response.outcome == "refusal") if not expected else is_answer
        return EvaluationOutcome(
            retrieval_success=retrieval_success,
            grounded=grounded,
            citation_correct=citation_correct,
            answer_quality=answer_quality,
            refusal_correct=refusal_correct,
            response_time_ms=float(response.trace.timings_ms.get("total_ms", 0)),
        )


def main() -> None:
    project = _required("GOOGLE_CLOUD_PROJECT")
    run_id = _required("EVALUATION_RUN_ID")
    client = firestore.Client(project=project)
    datasets = FirestoreEvaluationGenerationStore(client)
    runs = FirestoreEvaluationRunStore(client)
    run = runs.get(run_id)
    if run is None:
        raise EvaluationRunError("unknown evaluation run")
    try:
        dataset = datasets.get(run.dataset_version_id)
        if dataset is None:
            execute_run(run_id, datasets=datasets, runs=runs, evaluate=lambda _: _unavailable())
            return
        execute_run(
            run_id,
            datasets=datasets,
            runs=runs,
            evaluate=EvaluationAnswerEvaluator(
                dataset, _pinned_answerer(project, run.corpus_version_id)
            ),
        )
    except Exception:
        # Configuration and provider setup failures must be durable, but may
        # still exit non-zero so Cloud Run applies its bounded retry policy.
        runs.update(run.model_copy(update={"status": "failed", "error": "evaluation worker setup failed"}))
        raise


def _pinned_answerer(project: str, corpus_version_id: str) -> RetrievedAnswerService:
    client = firestore.Client(project=project)
    factory = LangChainRagFactory.from_environment(client)
    corpus = FirestoreCorpusVersionStore(
        client, vector_collection=factory.settings.vector_collection
    ).get(corpus_version_id)
    if corpus is None:
        raise EvaluationRunError("run references an unavailable corpus version")
    adapter = factory.generation_adapter()

    class _PinnedCorpus:
        def active(self):
            return corpus

    request_timeout_ms, generation_timeout_ms = timeout_settings(
        os.environ.get("ANSWER_REQUEST_TIMEOUT_MS", str(DEFAULT_REQUEST_TIMEOUT_MS)),
        os.environ.get("ANSWER_GENERATION_TIMEOUT_MS", str(DEFAULT_GENERATION_TIMEOUT_MS)),
    )
    return RetrievedAnswerService(
        factory.retriever(),
        _PinnedCorpus(),
        GroundedAnswerPolicy(
            adapter,
            model_name=factory.selected_model(),
            timeout_ms=generation_timeout_ms,
        ),
        request_timeout_ms=request_timeout_ms,
        retriever_for_binding=lambda pinned: factory.retriever_for_binding(
            pinned.serving_binding
            or legacy_serving_binding(
                pinned, vector_collection=factory.settings.vector_collection
            )
        ),
    )


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
