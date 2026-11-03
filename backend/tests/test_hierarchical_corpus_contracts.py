import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    ChildDocumentChunk,
    ParentChunk,
    RetrievalResult,
    SelectedParent,
    SourceSpan,
    StructureMap,
    StructureMapEntry,
)
from app.corpus.tokenization import TOKENIZER_ENCODING, TOKENIZER_PACKAGE_VERSION, count_tokens


FIXTURE = json.loads((Path(__file__).parent / "fixtures/corpus/hierarchical-contract.json").read_text())


def _parent() -> ParentChunk:
    return ParentChunk(
        parent_id=FIXTURE["parent_id"],
        corpus_version_id=FIXTURE["corpus_version_id"],
        section_path=FIXTURE["section_path"],
        page_start=10,
        page_end=11,
        source_spans=FIXTURE["spans"],
        content_hash=FIXTURE["content_hash"],
        token_count=12,
        text="1. Accountability\\n- owner\\n| role | duty |",
    )


def test_hierarchical_contract_round_trip_preserves_page_linked_provenance() -> None:
    parent = _parent()
    first_child = ChildDocumentChunk(
        chunk_id="v2:child:0",
        corpus_version_id="v2",
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=10,
        page_end=11,
        source_spans=parent.source_spans,
        child_ordinal=0,
        content_hash="b" * 64,
        token_count=8,
        text="Accountability owner role duty",
    )

    second_child = first_child.model_copy(update={"chunk_id": "v2:child:1", "child_ordinal": 1})
    restored = ChildDocumentChunk.model_validate_json(first_child.model_dump_json())

    assert restored.section_path == ("Data Governance", "Accountability")
    assert (restored.page_start, restored.page_end) == (10, 11)
    assert restored.source_spans == parent.source_spans
    assert (restored.parent_id, restored.chunk_id, restored.content_hash) == ("v2:parent:0", "v2:child:0", "b" * 64)
    assert {first_child.parent_id, second_child.parent_id} == {parent.parent_id}

    structure_map = StructureMap(
        map_id="reviewed-map-v1",
        source_sha256="c" * 64,
        content_hash="d" * 64,
        entries=[StructureMapEntry(section_path=FIXTURE["unclassified_section_path"], page_start=12, page_end=12)],
    )
    assert structure_map.entries[0].section_path[-1] == "Unclassified"


def test_child_contract_rejects_flat_chunk_and_unordered_source_spans() -> None:
    with pytest.raises(ValidationError, match="parent_id"):
        ChildDocumentChunk.model_validate({"chunk_id": "flat"})
    with pytest.raises(ValueError, match="ordered"):
        ParentChunk(
            **_parent().model_dump(exclude={"source_spans"}),
            source_spans=[
                SourceSpan(page=11, start_offset=0, end_offset=1, text="b"),
                SourceSpan(page=10, start_offset=0, end_offset=1, text="a"),
            ],
        )


def test_pinned_token_counter_is_repeatable() -> None:
    text = "A table row and a complete list item."
    assert TOKENIZER_ENCODING == "cl100k_base"
    assert TOKENIZER_PACKAGE_VERSION == "0.12.0"
    assert count_tokens(text) == count_tokens(text) == 9


def test_retrieval_selected_parents_are_bounded_and_ranked() -> None:
    result = RetrievalResult.model_validate({
        "candidate": {"artifact_id": "candidate", "installed_lock_sha256": "c" * 64, "corpus_version_id": "v2", "configuration_version_id": "cfg", "scorer_version": "v1"},
        "score_semantics": "cosine_similarity", "strategy": "dense", "stage_status": "succeeded",
        "selected_parents": [SelectedParent(parent_id="v2:parent:0", token_count=12, rank=0)],
    })
    assert result.selected_parents[0].parent_id == "v2:parent:0"
