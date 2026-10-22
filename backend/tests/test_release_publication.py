from datetime import datetime, timezone

import pytest

from app.contracts import CorpusServingBinding, CorpusValidationEvidence, CorpusVersion, ReleaseDecision
from app.release import (
    ConcurrentPublicationError,
    InMemoryServingReleaseStore,
    PublicationError,
    ReleaseManifest,
    ReleasePublisher,
)


def corpus(status: str = "validated") -> CorpusVersion:
    return CorpusVersion(
        version_id="corpus-v2",
        source_uri="gs://corpus/source.pdf#2",
        embedding_model="gemini-embedding-001",
        status=status,
        created_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
        serving_binding=CorpusServingBinding(
            corpus_version_id="corpus-v2",
            vector_collection="chunks-v2",
            vector_schema_version="2",
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
        ),
        validation_evidence=CorpusValidationEvidence(
            chunk_count=1,
            content_hashes_sha256="c" * 64,
            provenance_sha256="d" * 64,
            index_ready=True,
            filter_ready=True,
        ),
    )


def decision() -> ReleaseDecision:
    return ReleaseDecision(
        release_id="release-v2", evaluation_run_id="run-v2", dataset_version_id="dataset-v2",
        corpus_version_id="corpus-v2", configuration_version_id="config-v2",
        provider_version_id="openrouter-v1", model_version_id="model-v2",
        scorer_version_id="scorer-v2", gate_report_ids=["gate-v2"], decision="approved",
        rationale="all gates passed",
    )


def manifest() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="release-v2", evaluation_run_id="run-v2", dataset_version_id="dataset-v2",
        corpus_version_id="corpus-v2", configuration_version_id="config-v2",
        provider_version_id="openrouter-v1", model_version_id="model-v2",
        scorer_version_id="scorer-v2", candidate_digest="e" * 64, gate_report_ids=["gate-v2"],
    )


def test_publication_is_the_only_serving_transition_and_preserves_tuple() -> None:
    store = InMemoryServingReleaseStore()
    published = ReleasePublisher(store, clock=lambda: datetime(2026, 10, 22, tzinfo=timezone.utc)).publish(
        decision(), manifest(), corpus(), expected_previous_release_id=None
    )

    assert store.current() == published
    assert published.corpus_version_id == "corpus-v2"
    assert published.vector_collection == "chunks-v2"
    assert published.source_uri.endswith("#2")


def test_compare_and_set_has_one_winner_and_rejects_stale_previous_release() -> None:
    store = InMemoryServingReleaseStore()
    publisher = ReleasePublisher(store)
    first = publisher.publish(decision(), manifest(), corpus(), expected_previous_release_id=None)

    next_corpus = corpus().model_copy(update={"version_id": "corpus-v3"})
    next_corpus = next_corpus.model_copy(update={
        "serving_binding": next_corpus.serving_binding.model_copy(update={"corpus_version_id": "corpus-v3"}),
    })
    next_decision = decision().model_copy(update={"release_id": "release-v3", "corpus_version_id": "corpus-v3"})
    next_manifest = manifest().model_copy(update={"release_id": "release-v3", "corpus_version_id": "corpus-v3"})
    second = publisher.publish(next_decision, next_manifest, next_corpus, expected_previous_release_id=first.release_id)

    with pytest.raises(ConcurrentPublicationError):
        publisher.publish(decision(), manifest(), corpus(), expected_previous_release_id=None)

    assert store.previous() == first
    assert store.rollback(expected_current_release_id=second.release_id) == first


def test_publication_rejects_unready_or_unapproved_candidates() -> None:
    publisher = ReleasePublisher(InMemoryServingReleaseStore())
    with pytest.raises(PublicationError):
        publisher.publish(decision().model_copy(update={"decision": "rejected"}), manifest(), corpus(), expected_previous_release_id=None)
    with pytest.raises(PublicationError):
        publisher.publish(decision(), manifest(), corpus("staged"), expected_previous_release_id=None)
