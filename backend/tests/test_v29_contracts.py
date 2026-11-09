from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.contracts.v29 import (
    AllowanceReservation,
    CandidateDatasetBinding,
    CorpusStagingAttemptV2,
    DatasetRevisionConflict,
    EvaluationRowV2,
    EvaluationRunSnapshot,
    GenerationContextParent,
    ParsedBlock,
    ParsingArtifact,
    RawSourceSpan,
    ServingDatasetBinding,
)

DIGEST = "a" * 64


def span():
    return RawSourceSpan(span_id="s1", block_id="b1", physical_page=4, printed_page_label="ii", start_offset=0, end_offset=4, raw_text="text")


def refusal():
    return EvaluationRowV2(item_id="i1", origin="generated", question="Outside scope?", category="study", expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse", expected_refusal=True, scope_rationale="Outside corpus")


def test_artifact_distinguishes_physical_and_printed_pages_and_orders_blocks():
    block = ParsedBlock(block_id="b1", physical_page=4, printed_page_label="ii", reading_order=0, kind="text", raw_text="text")
    artifact = ParsingArtifact(source_generation="g1", source_sha256=DIGEST, output_sha256=DIGEST, blocks=[block])
    assert artifact.blocks[0].physical_page == 4
    assert artifact.blocks[0].printed_page_label == "ii"
    with pytest.raises(ValidationError):
        ParsingArtifact(source_generation="g1", source_sha256=DIGEST, output_sha256=DIGEST, blocks=[block, block])


def test_context_keeps_raw_spans_separate_from_embedding_text():
    parent = GenerationContextParent(parent_id="p1", spans=[span()], backing_child_ids=["c1"], embedding_text="Breadcrumb: text")
    assert parent.spans[0].raw_text == "text"
    assert parent.embedding_text == "Breadcrumb: text"


def test_refusal_requires_zero_support_and_answer_requires_bindings():
    assert refusal().relevant_child_ids == ()
    with pytest.raises(ValidationError):
        EvaluationRowV2.model_validate({**refusal().model_dump(), "relevant_child_ids": ["c1"]})
    with pytest.raises(ValidationError):
        EvaluationRowV2.model_validate({**refusal().model_dump(), "expected_outcome": "answer", "expected_refusal": False})


def test_revision_conflict_and_binding_types_are_explicit():
    assert DatasetRevisionConflict(expected_revision=1, current_revision=2, message="Refresh").current_revision == 2
    with pytest.raises(ValidationError):
        DatasetRevisionConflict(expected_revision=0, current_revision=2, message="Refresh")
    with pytest.raises(ValidationError):
        CandidateDatasetBinding.model_validate(ServingDatasetBinding(release_id="r", corpus_version_id="c", dataset_revision=1).model_dump())
    with pytest.raises(ValidationError):
        EvaluationRunSnapshot(run_id="run", binding={"kind": "candidate", "release_id": "r", "corpus_version_id": "c", "dataset_revision": 1}, configuration_version_id="cfg", selection="full", items=[refusal()], snapshot_sha256=DIGEST)


def test_artifact_pair_and_allowance_expiry_are_validated():
    with pytest.raises(ValidationError):
        CorpusStagingAttemptV2(attempt_id="a", actor_id="actor", source_generation="g", source_sha256=DIGEST, parser_route="route", parser_settings_sha256=DIGEST, parsing_artifact_id="artifact", status="parsing")
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        AllowanceReservation(reservation_id="r", operation_key="o", kind="parsing", allowance_observed_at=now, allowance_expires_at=now, reserved_units=1)


def test_shared_contract_fixture():
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).resolve().parents[2] / "documents/contracts/v29-contract-fixture.json").read_text())
    assert RawSourceSpan.model_validate(fixture["span"]).physical_page == 4
    assert fixture["span"]["raw_text"] != fixture["embedding_text"]
    assert EvaluationRowV2.model_validate(fixture["refusal"]).expected_refusal
    assert DatasetRevisionConflict.model_validate(fixture["revision_conflict"]).current_revision == 2


def test_structured_blocks_and_synthetic_mapping_keep_source_boundaries():
    from app.contracts.v29 import SourceTextMapping

    table = ParsedBlock(block_id="t", physical_page=1, reading_order=0, kind="table", raw_text="a b", table_rows=(("a", "b"),))
    assert table.table_rows == (("a", "b"),)
    with pytest.raises(ValidationError):
        ParsedBlock(block_id="t", physical_page=1, reading_order=0, kind="table", raw_text="a b")
    mapping = SourceTextMapping(target_start=0, target_end=4, source_span_id="s1", source_start=0, source_end=4)
    synthetic = SourceTextMapping(target_start=4, target_end=5, synthetic=True)
    GenerationContextParent(parent_id="p", spans=[span()], backing_child_ids=["c"], embedding_text="text!", normalization_mappings=[mapping, synthetic])
    with pytest.raises(ValidationError):
        SourceTextMapping(target_start=0, target_end=1, synthetic=True, source_span_id="s1")


def test_raw_source_whitespace_and_offsets_are_preserved():
    raw = "  source text\n"
    block = ParsedBlock(block_id="b", physical_page=1, reading_order=0, kind="text", raw_text=raw)
    source = RawSourceSpan(span_id="s", block_id="b", physical_page=1, start_offset=5, end_offset=5 + len(raw), raw_text=raw)
    assert block.raw_text == source.raw_text == raw
    assert source.end_offset - source.start_offset == len(raw)
