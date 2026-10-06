from datetime import datetime, timezone

import pytest

from app.contracts import CorpusVersion, DocumentChunk
from app.corpus import CorpusCompletenessError, InMemoryCorpusVersionStore


def version(version_id: str, status: str = "staged") -> CorpusVersion:
    return CorpusVersion(
        version_id=version_id,
        source_uri=f"gs://corpus/{version_id}.pdf#1",
        embedding_model="gemini-embedding-001",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def chunk(version_id: str, ordinal: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{version_id}:1:{ordinal}",
        corpus_version_id=version_id,
        page=1,
        section="Body",
        chunk_ordinal=ordinal,
        content_hash="a" * 64,
        text="A passage",
        embedding=[0.0] * 768,
    )


def test_invalid_corpus_cannot_activate_and_pointer_is_unchanged() -> None:
    store = InMemoryCorpusVersionStore()
    store.stage(version("v1"))
    store.validate("v1", [chunk("v1")])
    store.activate("v1")

    store.stage(version("v2"))
    with pytest.raises(CorpusCompletenessError):
        store.validate("v2", [])

    assert store.active().version_id == "v1"


def test_activation_is_idempotent_and_rollback_restores_previous_pointer() -> None:
    store = InMemoryCorpusVersionStore()
    for version_id in ("v1", "v2"):
        store.stage(version(version_id))
        store.validate(version_id, [chunk(version_id)])
        store.activate(version_id)

    activated = store.activate("v2")
    restored = store.rollback()

    assert activated.version_id == "v2"
    assert restored.version_id == "v1"
    assert restored.status == "active"
    assert store.active().version_id == "v1"
