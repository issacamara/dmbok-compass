from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.contracts import CorpusServingBinding, CorpusValidationEvidence, CorpusVersion, ReleaseDecision
from app.release import (
    ConcurrentPublicationError,
    FirestoreServingReleaseStore,
    InMemoryServingReleaseStore,
    PublicationError,
    PublishedRelease,
    ReleaseCandidateBinding,
    ReleaseManifest,
    ReleasePublisher,
    release_candidate_digest,
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
    binding = candidate_binding()
    return ReleaseDecision(
        release_id="release-v2", evaluation_run_id="run-v2", dataset_version_id="dataset-v2",
        corpus_version_id="corpus-v2", configuration_version_id="config-v2",
        provider_version_id="openrouter-v1", model_version_id="model-v2",
        scorer_version_id="scorer-v2", gate_report_ids=["gate-v2"], decision="approved",
        candidate_digest=release_candidate_digest(binding), rationale="all gates passed",
    )


def candidate_binding(**overrides: object) -> ReleaseCandidateBinding:
    values: dict[str, object] = {
        "evaluation_run_id": "run-v2", "dataset_version_id": "dataset-v2", "dataset_revision": 4,
        "dataset_snapshot_sha256": "1" * 64, "dataset_revision_sha256": "2" * 64,
        "corpus_version_id": "corpus-v2",
        "corpus_manifest_sha256": "a" * 64, "corpus_configuration_sha256": "b" * 64,
        "candidate_snapshot_sha256": "c" * 64, "generator_sha256": "d" * 64,
        "judge_version": "scorer-v2", "output_assessment_sha256": "e" * 64,
        "context_policy_sha256": "f" * 64, "dependency_lock_sha256": "0" * 64,
        "integration_artifact": "application-image@sha256:" + "1" * 64,
        "runtime_version": "image@sha256:" + "2" * 64,
    }
    values.update(overrides)
    return ReleaseCandidateBinding(**values)


def manifest() -> ReleaseManifest:
    binding = candidate_binding()
    return ReleaseManifest(
        release_id="release-v2", evaluation_run_id="run-v2", dataset_version_id="dataset-v2",
        corpus_version_id="corpus-v2", configuration_version_id="config-v2",
        provider_version_id="openrouter-v1", model_version_id="model-v2",
        scorer_version_id="scorer-v2", candidate_digest=release_candidate_digest(binding),
        gate_report_ids=["gate-v2"], candidate_binding=binding,
    )


def serving_store() -> InMemoryServingReleaseStore:
    store = InMemoryServingReleaseStore()
    store.set_dataset_head("corpus-v2", revision=4, dataset_sha256="2" * 64)
    return store


def test_publication_is_the_only_serving_transition_and_preserves_tuple() -> None:
    store = serving_store()
    published = ReleasePublisher(store, clock=lambda: datetime(2026, 10, 22, tzinfo=timezone.utc)).publish(
        decision(), manifest(), corpus(), expected_previous_release_id=None
    )

    assert store.current() == published
    assert published.corpus_version_id == "corpus-v2"
    assert published.vector_collection == "chunks-v2"
    assert published.source_uri.endswith("#2")


def test_compare_and_set_has_one_winner_and_rejects_stale_previous_release() -> None:
    store = serving_store()
    publisher = ReleasePublisher(store)
    first = publisher.publish(decision(), manifest(), corpus(), expected_previous_release_id=None)

    next_corpus = corpus().model_copy(update={"version_id": "corpus-v3"})
    next_corpus = next_corpus.model_copy(update={
        "serving_binding": next_corpus.serving_binding.model_copy(update={"corpus_version_id": "corpus-v3"}),
    })
    next_binding = candidate_binding(corpus_version_id="corpus-v3")
    store.set_dataset_head(
        "corpus-v3", revision=next_binding.dataset_revision,
        dataset_sha256=next_binding.dataset_revision_sha256,
    )
    next_digest = release_candidate_digest(next_binding)
    next_decision = decision().model_copy(update={
        "release_id": "release-v3", "corpus_version_id": "corpus-v3", "candidate_digest": next_digest,
    })
    next_manifest = manifest().model_copy(update={
        "release_id": "release-v3", "corpus_version_id": "corpus-v3",
        "candidate_binding": next_binding, "candidate_digest": next_digest,
    })
    second = publisher.publish(next_decision, next_manifest, next_corpus, expected_previous_release_id=first.release_id)

    with pytest.raises(ConcurrentPublicationError):
        publisher.publish(decision(), manifest(), corpus(), expected_previous_release_id=None)

    assert store.previous() == first
    assert store.rollback(expected_current_release_id=second.release_id) == first


def test_publication_rejects_unready_or_unapproved_candidates() -> None:
    publisher = ReleasePublisher(serving_store())
    with pytest.raises(PublicationError):
        publisher.publish(decision().model_copy(update={"decision": "rejected"}), manifest(), corpus(), expected_previous_release_id=None)
    with pytest.raises(PublicationError):
        publisher.publish(decision(), manifest(), corpus("staged"), expected_previous_release_id=None)


def test_publication_rejects_wrong_candidate_tuple_and_stale_decision() -> None:
    publisher = ReleasePublisher(serving_store())
    wrong_manifest_binding = candidate_binding(corpus_manifest_sha256="9" * 64)
    wrong_manifest = manifest().model_copy(update={
        "candidate_binding": wrong_manifest_binding,
        "candidate_digest": release_candidate_digest(wrong_manifest_binding),
    })
    with pytest.raises(PublicationError, match="exact reviewed candidate"):
        publisher.publish(decision(), wrong_manifest, corpus(), expected_previous_release_id=None)

    stale_decision = decision().model_copy(update={"candidate_digest": "9" * 64})
    with pytest.raises(PublicationError, match="exact reviewed candidate"):
        publisher.publish(stale_decision, manifest(), corpus(), expected_previous_release_id=None)


def test_manifest_candidate_digest_covers_dataset_generator_judge_assessment_and_context() -> None:
    original = candidate_binding()
    original_digest = release_candidate_digest(original)
    for field, replacement in (
        ("dataset_revision", 5), ("generator_sha256", "9" * 64),
        ("judge_version", "scorer-v3"), ("output_assessment_sha256", "8" * 64),
        ("context_policy_sha256", "7" * 64), ("dependency_lock_sha256", "6" * 64),
        ("integration_artifact", "application-image@sha256:" + "5" * 64),
        ("corpus_manifest_sha256", "4" * 64),
    ):
        assert release_candidate_digest(original.model_copy(update={field: replacement})) != original_digest


def test_publish_endpoint_revalidates_approved_tuple_before_switching_serving(monkeypatch) -> None:
    import firebase_admin.firestore

    from app import main
    decision_record = decision()
    store = serving_store()

    class DecisionService:
        decisions = SimpleNamespace(get=lambda release_id: decision_record)
        manifests = SimpleNamespace(get=lambda release_id: manifest())

        def record(self, request):
            assert request.evaluation_run_id == decision_record.evaluation_run_id
            # Revalidation observed the approved revision. A dataset edit lands
            # after that read but before the serving transaction starts.
            assert store.dataset_head("corpus-v2") == (4, "2" * 64)
            store.set_dataset_head("corpus-v2", revision=5, dataset_sha256="9" * 64)
            return decision_record

    monkeypatch.setattr("app.identity._firebase_app", lambda: object())
    monkeypatch.setattr(firebase_admin.firestore, "client", lambda *, app: object())
    monkeypatch.setattr(main, "startup_configuration", lambda: SimpleNamespace(vector_collection="chunks-v2"))
    monkeypatch.setattr(
        main, "FirestoreCorpusVersionStore",
        lambda client, *, vector_collection: SimpleNamespace(get=lambda corpus_id: corpus()),
    )
    monkeypatch.setattr(main, "FirestoreServingReleaseStore", lambda client: store)

    with pytest.raises(HTTPException) as error:
        main.publish_release("release-v2", None, decision_service=DecisionService())

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "release_not_publishable"
    assert store.current() is None


def test_firestore_publication_transaction_fences_dataset_head(monkeypatch) -> None:
    from hashlib import sha256
    from google.cloud import firestore

    from test_evaluation_generations import FakeFirestore

    client = FakeFirestore()
    head_id = sha256("corpus-v2".encode()).hexdigest()
    client.data[f"evaluation_dataset_heads_v2/{head_id}"] = {
        "corpus_version_id": "corpus-v2", "revision": 5, "dataset_sha256": "9" * 64,
    }

    def transactional(function):
        return lambda transaction: function(transaction)

    monkeypatch.setattr(firestore, "transactional", transactional)
    store = FirestoreServingReleaseStore(client)
    release = PublishedRelease(
        release_id="release-v2", artifact_id="artifact", configuration_version_id="config-v2",
        corpus_version_id="corpus-v2", vector_collection="chunks-v2", vector_schema_version="2",
        source_uri="gs://corpus/source.pdf", published_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )

    with pytest.raises(ConcurrentPublicationError, match="dataset revision changed"):
        store.publish(
            release, expected_previous_release_id=None,
            expected_dataset_revision=4, expected_dataset_sha256="2" * 64,
        )

    assert "release_state/serving" not in client.data
