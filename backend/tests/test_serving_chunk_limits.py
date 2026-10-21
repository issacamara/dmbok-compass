from __future__ import annotations

from copy import deepcopy

import pytest

from app.contracts import RetrievedPassage
from app.corpus import SourcePage, chunk_pages, create_source_manifest, default_derivation
from app.corpus.chunking import MAX_CHUNK_CHARACTERS


def test_oversized_long_word_is_subdivided_without_content_loss_and_round_trips() -> None:
    text = "x" * 7_699

    chunks = chunk_pages(
        [SourcePage(page=4, section="Long word", text=text)],
        corpus_version_id="v-long-word",
    )

    assert [len(chunk.text) for chunk in chunks] == [MAX_CHUNK_CHARACTERS, 2_699]
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= MAX_CHUNK_CHARACTERS for chunk in chunks)
    for chunk in chunks:
        RetrievedPassage(
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            section=chunk.section,
            excerpt=chunk.text,
        )


def test_subdivision_ids_are_stable_for_retries_and_partial_batches() -> None:
    page = SourcePage(page=8, section="Stable", text="y" * 6_001)

    first = chunk_pages([page], corpus_version_id="v-stable")
    retry = chunk_pages([page], corpus_version_id="v-stable")

    assert first == retry
    assert [chunk.chunk_id for chunk in first] == ["v-stable:8:0", "v-stable:8:1"]
    assert [chunk.content_hash for chunk in first] == [chunk.content_hash for chunk in retry]


def test_subdivision_algorithm_is_in_manifest_identity_and_invalidates_old_evidence() -> None:
    source = {
        "bucket": "dmbok-corpus",
        "object_name": "source/dmbok.pdf",
        "generation": 7,
        "content_sha256": "a" * 64,
    }
    original = create_source_manifest(**source)
    changed_derivation = deepcopy(default_derivation())
    changed_derivation["splitter"]["subdivision"]["max_characters"] = 4_000
    changed = create_source_manifest(**source, derivation=changed_derivation)

    assert original.corpus_version_id != changed.corpus_version_id
    assert original.derivation_sha256 != changed.derivation_sha256


def test_chunking_rejects_provenance_that_cannot_fit_the_serving_contract() -> None:
    with pytest.raises(ValueError, match="section"):
        chunk_pages(
            [SourcePage(page=1, section="s" * 501, text="text")],
            corpus_version_id="v-1",
        )
