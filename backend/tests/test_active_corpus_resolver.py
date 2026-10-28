from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from resolve_active_corpus import CorpusResolutionError, resolve_active_corpus


class Snapshot:
    def __init__(self, data):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return self._data


class Document:
    def __init__(self, data):
        self.data = data

    def get(self):
        return Snapshot(self.data)


class Collection:
    def __init__(self, documents):
        self.documents = documents

    def document(self, name):
        return Document(self.documents.get(name))


class Client:
    def __init__(self, pointer, version):
        self.collections = {
            "corpus_state": {"active": pointer},
            "corpus_versions": {"v-ready": version} if version is not None else {},
        }

    def collection(self, name):
        return Collection(self.collections[name])


def valid_version():
    binding = {
        "corpus_version_id": "v-ready",
        "vector_collection": "document_chunks_v2",
        "vector_schema_version": "2",
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        "ingestion_fingerprint": "a" * 64,
        "configuration_fingerprint": "b" * 64,
    }
    return {
        "version_id": "v-ready",
        "source_uri": "gs://bucket/corpus.pdf",
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "serving_binding": binding,
        "validation_evidence": {
            "chunk_count": 1,
            "content_hashes_sha256": "c" * 64,
            "provenance_sha256": "d" * 64,
            "embedding_dimensions": 768,
            "index_ready": True,
            "filter_ready": True,
        },
    }


def client_with(**changes):
    version = valid_version()
    pointer = {"version_id": "v-ready", "binding": version["serving_binding"]}
    version.update(changes)
    return Client(pointer, version)


def test_resolves_valid_active_ready_corpus():
    assert resolve_active_corpus(client_with()) == "v-ready"


@pytest.mark.parametrize(
    "pointer",
    [None, {}, {"version_id": ""}, {"version_id": "v-ready", "binding": "invalid"}, {"version_id": "v-ready", "binding": {"corpus_version_id": "other"}}],
    ids=["absent", "malformed", "empty-id", "malformed-binding", "mismatched-binding"],
)
def test_rejects_absent_or_malformed_pointer(pointer):
    with pytest.raises(CorpusResolutionError):
        resolve_active_corpus(Client(pointer, valid_version()))


def test_rejects_missing_version():
    with pytest.raises(CorpusResolutionError):
        resolve_active_corpus(Client({"version_id": "v-ready"}, None))


def test_accepts_legacy_pointer_without_optional_binding():
    assert resolve_active_corpus(Client({"version_id": "v-ready"}, valid_version())) == "v-ready"


def test_accepts_legacy_version_without_persisted_serving_binding():
    version = valid_version()
    version["serving_binding"] = None
    assert resolve_active_corpus(Client({"version_id": "v-ready"}, version)) == "v-ready"


def test_rejects_non_active_version():
    with pytest.raises(CorpusResolutionError):
        resolve_active_corpus(client_with(status="validated"))


def test_rejects_mismatched_serving_binding():
    version = valid_version()
    pointer = {"version_id": "v-ready", "binding": {**version["serving_binding"], "corpus_version_id": "other"}}
    with pytest.raises(CorpusResolutionError):
        resolve_active_corpus(Client(pointer, version))


@pytest.mark.parametrize(
    "evidence",
    [None, {"chunk_count": 0, "content_hashes_sha256": "c" * 64, "provenance_sha256": "d" * 64, "index_ready": True, "filter_ready": True},
        {"chunk_count": 1, "content_hashes_sha256": "c" * 64, "provenance_sha256": "d" * 64, "index_ready": False, "filter_ready": True},
        {"chunk_count": 1, "content_hashes_sha256": "c" * 64, "provenance_sha256": "d" * 64, "index_ready": True, "filter_ready": False}],
    ids=["absent", "zero-chunks", "index-not-ready", "filter-not-ready"],
)
def test_rejects_missing_or_unready_evidence(evidence):
    with pytest.raises(CorpusResolutionError):
        resolve_active_corpus(client_with(validation_evidence=evidence))
