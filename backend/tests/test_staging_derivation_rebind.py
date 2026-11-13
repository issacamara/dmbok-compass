from copy import deepcopy

import pytest

from app.corpus.firestore import FirestoreCorpusVersionStore
from app.corpus.staging import FirestoreCorpusStagingAttemptStore, StagingAttemptConflict
from app.corpus.versions import CorpusVersionError
from test_evaluation_generations import FakeCollection, FakeFirestore, FakeSnapshot, run_fake_transaction, _field_value


class Query:
    def __init__(self, collection, field, value):
        self.collection, self.field, self.value = collection, field, value

    def limit(self, count):
        assert count == 1
        return self

    def stream(self, *, transaction):
        prefix = self.collection.path + "/"
        depth = self.collection.path.count("/") + 1
        return iter([FakeSnapshot(data) for path, data in transaction.data.items()
                     if path.startswith(prefix) and path.count("/") == depth
                     and _field_value(data, self.field) == self.value][:1])


@pytest.fixture
def setup_retry(monkeypatch):
    monkeypatch.setattr(FakeCollection, "where", lambda self, *, filter: Query(self, filter.field_path, filter.value))
    db = FakeFirestore()
    source = {"bucket": "bucket", "object_name": "staging/a/source.pdf", "generation": 1}
    map_artifact = {"object_name": "approved-map", "generation": 2}
    approval_artifact = {"object_name": "approval", "generation": 3}
    attempt = {
        "attempt_id": "a", "revision": 4, "state": "failed", "completed_stage": "ingestion",
        "error": {"code": "job_failed", "message": "failed", "status": 503, "retryable": True},
        "correction_codes": ["job_failed"], "dispatch_retryable": True, "dispatch_claimed": False,
        "source": source, "structure": {"object_name": "structure"},
        "approved_map_artifact": map_artifact, "approval_artifact": approval_artifact,
        "staged_corpus_version_id": "old", "parent_count": 0, "child_count": 0,
    }
    db.data["corpus_staging_attempts/a"] = deepcopy(attempt)
    for version_id in ("old", "new"):
        db.data[f"corpus_versions/{version_id}"] = {
            "version_id": version_id, "status": "staged", "source_uri": "gs://bucket/source#1",
            "serving_binding": {"corpus_version_id": version_id, "vector_collection": "vectors"},
        }
    repository = FirestoreCorpusStagingAttemptStore(db, transaction_runner=run_fake_transaction)
    approval = {"_map_artifact": map_artifact, "_approval_artifact": approval_artifact}
    queued, dispatched = repository.claim_approval(attempt_id="a", expected_revision=4, approval=approval)
    assert queued.state == "queued" and dispatched
    return db, repository, source, map_artifact, approval_artifact


def rebind(setup):
    db, store, source, map_artifact, approval_artifact = setup
    return store.record_ingestion_progress(attempt_id="a", source=source, map_artifact=map_artifact,
                                           approval_artifact=approval_artifact, state="running",
                                           parent_count=0, child_count=0, corpus_version_id="new")


def test_retry_rebinds_only_empty_old_derivation_and_preserves_history(setup_retry):
    db, _, source, map_artifact, approval = setup_retry
    original = deepcopy(db.data["corpus_versions/old"])
    db.data["corpus_ingestion_runs/old"] = {"status": "failed", "updated_at": "before", "audit": "retained"}
    status = rebind(setup_retry)
    attempt = db.data["corpus_staging_attempts/a"]
    assert status.staged_corpus_version_id == "new"
    assert attempt["previous_corpus_version_ids"] == ["old"]
    assert (attempt["source"], attempt["approved_map_artifact"], attempt["approval_artifact"]) == (source, map_artifact, approval)
    assert db.data["corpus_versions/old"] == original
    assert db.data["corpus_ingestion_runs/old"]["status"] == "superseded"
    assert db.data["corpus_ingestion_runs/old"]["previous_status"] == "failed"
    assert db.data["corpus_ingestion_runs/old"]["previous_updated_at"] == "before"
    assert db.data["corpus_ingestion_runs/old"]["audit"] == "retained"
    versions = FirestoreCorpusVersionStore(db, transaction_runner=run_fake_transaction)
    with pytest.raises(CorpusVersionError, match="superseded"):
        versions.begin_staging("old")
    with pytest.raises(CorpusVersionError, match="superseded"):
        versions.cleanup_staging("old")


@pytest.mark.parametrize("path,payload", [
    ("corpus_parent_chunks/p", {"corpus_version_id": "old"}),
    ("vectors/c", {"metadata": {"corpus_version_id": "old"}}),
    ("corpus_ingestion_runs/old", {"status": "running"}),
])
def test_retry_refuses_existing_records_or_active_writer(setup_retry, path, payload):
    db, _, _, _, _ = setup_retry
    db.data[path] = payload
    with pytest.raises(StagingAttemptConflict, match="cannot safely be rebound"):
        rebind(setup_retry)
    assert db.data["corpus_staging_attempts/a"]["staged_corpus_version_id"] == "old"
    assert db.data[path] == payload


def test_rebind_rejects_changed_approval_or_unapproved_retry(setup_retry):
    db, store, source, map_artifact, approval = setup_retry
    with pytest.raises(StagingAttemptConflict, match="artifacts do not match"):
        store.record_ingestion_progress(attempt_id="a", source=source,
                                        map_artifact={**map_artifact, "generation": 99},
                                        approval_artifact=approval, state="running",
                                        parent_count=0, child_count=0, corpus_version_id="new")
    assert "corpus_ingestion_runs/old" not in db.data
    db.data["corpus_staging_attempts/a"]["derivation_rebind_permitted"] = False
    with pytest.raises(StagingAttemptConflict, match="different corpus version"):
        rebind(setup_retry)
    assert db.data["corpus_staging_attempts/a"]["staged_corpus_version_id"] == "old"


def test_rebind_rejects_changed_source_or_existing_progress(setup_retry):
    db, _, _, _, _ = setup_retry
    db.data["corpus_versions/new"]["source_uri"] = "gs://bucket/other#1"
    with pytest.raises(StagingAttemptConflict, match="cannot safely be rebound"):
        rebind(setup_retry)
    db.data["corpus_versions/new"]["source_uri"] = db.data["corpus_versions/old"]["source_uri"]
    db.data["corpus_staging_attempts/a"]["parent_count"] = 1
    with pytest.raises(StagingAttemptConflict, match="different corpus version"):
        rebind(setup_retry)
    assert "corpus_ingestion_runs/old" not in db.data


def test_entrypoint_rebind_failure_stops_before_writer_and_keeps_old_identity(monkeypatch):
    from types import SimpleNamespace
    from app.ingestion import entrypoint

    events = []
    configuration = SimpleNamespace(runtime=SimpleNamespace(project="p", location="l"), vector_collection="vectors")
    monkeypatch.setattr(entrypoint.RuntimeConfiguration, "from_environment", staticmethod(lambda: configuration))
    monkeypatch.setattr(entrypoint.firestore, "Client", lambda **_: object())
    monkeypatch.setattr(entrypoint, "FirestoreCorpusVersionStore", lambda *_, **__: object())
    monkeypatch.setattr(entrypoint.storage, "Client", lambda **_: SimpleNamespace(bucket=lambda _: object()))
    monkeypatch.setattr(entrypoint, "_download_pinned_source", lambda _b, _o, path, **_: (path.write_bytes(b"pdf"), 1)[1])
    monkeypatch.setattr(entrypoint, "create_source_manifest", lambda **_: object())
    monkeypatch.setattr(entrypoint, "_read_pinned_object", lambda *_: (b"data", "gs://bucket/object#1", "a" * 64))
    monkeypatch.setattr(entrypoint, "build_and_stage_hierarchical_candidate", lambda **_: SimpleNamespace(
        version=SimpleNamespace(version_id="new"), children=(object(),),
    ))

    def progress(state, _parents, _children, version, _count, *_rest):
        events.append((state, version))
        if state == "running" and version == "new":
            raise StagingAttemptConflict("previous corpus cannot safely be rebound")

    monkeypatch.setattr(entrypoint, "_staging_progress_callback", lambda **_: progress)
    monkeypatch.setattr(entrypoint, "LangChainRagSettings", lambda **_: pytest.fail("writer must not start"))
    monkeypatch.setenv("CORPUS_STAGING_ATTEMPT_ID", "a")
    monkeypatch.setenv("CORPUS_BUCKET_NAME", "bucket")
    monkeypatch.setenv("CORPUS_OBJECT_NAME", "source")
    monkeypatch.delenv("CORPUS_CLEANUP_VERSION", raising=False)
    monkeypatch.delenv("CORPUS_PARSING_ARTIFACT_URI", raising=False)
    with pytest.raises(StagingAttemptConflict, match="cannot safely be rebound"):
        entrypoint._run(lambda: False)
    assert events == [("running", None), ("running", "new"), ("failed", None)]
