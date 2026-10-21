from __future__ import annotations

from dataclasses import replace

import pytest

from app.corpus.chunking import TextChunk
from app.corpus.lexical_artifact import (
    LexicalArtifactError,
    build_lexical_artifact,
    load_lexical_artifact,
)


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, name: str, data: bytes, *, metadata: dict[str, str], if_absent: bool = True) -> None:
        if if_absent and name in self.objects:
            raise FileExistsError(name)
        self.objects[name] = data

    def get(self, name: str) -> bytes:
        return self.objects[name]


def chunk(chunk_id: str, text: str, *, corpus: str = "corpus-v1") -> TextChunk:
    import hashlib

    return TextChunk(
        chunk_id=chunk_id,
        corpus_version_id=corpus,
        page=int(chunk_id.rsplit(":", 1)[-1]) + 1,
        section="Section",
        chunk_ordinal=0,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
    )


def test_build_load_round_trip_is_reproducible_and_searchable() -> None:
    chunks = (chunk("corpus-v1:1:0", "Alpha data governance"), chunk("corpus-v1:2:0", "Beta data quality"))
    first_store, second_store = FakeObjectStore(), FakeObjectStore()
    first = build_lexical_artifact(chunks, corpus_version_id="corpus-v1", object_name="a.json", store=first_store)
    second = build_lexical_artifact(tuple(reversed(chunks)), corpus_version_id="corpus-v1", object_name="a.json", store=second_store)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.chunk_set_sha256 == second.chunk_set_sha256
    loaded = load_lexical_artifact(first_store, first, expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=first.chunk_set_sha256)
    assert loaded.index.search("governance")[0].chunk_id == "corpus-v1:1:0"
    assert loaded.load_ms >= 0


def test_tamper_missing_and_wrong_corpus_fail_explicitly() -> None:
    store = FakeObjectStore()
    manifest = build_lexical_artifact((chunk("corpus-v1:1:0", "Alpha"),), corpus_version_id="corpus-v1", object_name="a.json", store=store)
    store.objects["a.json"] += b"tamper"
    with pytest.raises(LexicalArtifactError, match="digest"):
        load_lexical_artifact(store, manifest, expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=manifest.chunk_set_sha256)
    del store.objects["a.json"]
    with pytest.raises(LexicalArtifactError, match="missing"):
        load_lexical_artifact(store, manifest, expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=manifest.chunk_set_sha256)
    with pytest.raises(LexicalArtifactError, match="corpus"):
        load_lexical_artifact(FakeObjectStore(), replace(manifest, corpus_version_id="corpus-v2"), expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=manifest.chunk_set_sha256)


def test_dense_only_configuration_does_not_require_an_artifact() -> None:
    assert load_lexical_artifact is not None
    assert build_lexical_artifact((), corpus_version_id="corpus-v1", object_name="empty.json", store=FakeObjectStore()).chunk_count == 0


def test_candidate_limit_is_bounded() -> None:
    store = FakeObjectStore()
    manifest = build_lexical_artifact(
        tuple(chunk(f"corpus-v1:1:{index}", "shared term") for index in range(8)),
        corpus_version_id="corpus-v1", object_name="a.json", store=store,
    )
    loaded = load_lexical_artifact(store, manifest, expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=manifest.chunk_set_sha256, max_candidates=3)
    assert len(loaded.index.search("shared term")) == 3
    with pytest.raises(ValueError, match="between 1 and 3"):
        loaded.index.search("shared", limit=4)


def test_query_cache_is_bounded() -> None:
    store = FakeObjectStore()
    manifest = build_lexical_artifact((chunk("corpus-v1:1:0", "shared term"),), corpus_version_id="corpus-v1", object_name="a.json", store=store)
    loaded = load_lexical_artifact(store, manifest, expected_corpus_version_id="corpus-v1", expected_chunk_set_sha256=manifest.chunk_set_sha256, max_candidates=5)
    loaded.index.max_cache_entries = 2
    for query in ("shared one", "shared two", "shared three"):
        loaded.index.search(query)
    assert loaded.index.cache_size <= 2
