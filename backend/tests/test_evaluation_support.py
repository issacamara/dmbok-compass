from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from app.contracts import (
    ChildDocumentChunk,
    CorpusServingBinding,
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItem,
    EvaluationRun,
    GoldAnnotation,
    ParentChunk,
    SourceSpan,
)
from app.corpus.chunking import _token_count
from app.corpus.indexing import PARENT_COLLECTION
from app.corpus.versions import hierarchical_provenance_sha256, validate_hierarchical_records
from app.evaluation.job import EvaluationOutcome, EvaluationWorker, InMemoryEvaluationRunStore
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
from app.evaluation.batch_reviews import create_batch_review
from app.evaluation.support import (
    InMemorySupportEvidenceStore,
    FirestoreRunPinnedChunkSource,
    StoredSupportChunk,
    SupportPreparationError,
    prepare_support,
    review_support,
)


def snapshot() -> EvaluationCandidateSnapshot:
    return EvaluationCandidateSnapshot(
        corpus_binding=CorpusServingBinding(
            corpus_version_id="corpus-v1",
            vector_collection="document_chunks_v2",
            vector_schema_version="2",
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
        ),
        project="project",
        location="europe-west1",
        answer_provider="openrouter",
        primary_model="primary",
        fallback_model=None,
        json_schema_supported=True,
        request_timeout_ms=1000,
        generation_timeout_ms=1000,
        prompt_version="prompt-v1",
        retrieval_policy_version="retrieval-v1",
        evidence_policy_version="evidence-v1",
        scorer_version="scorer-v1",
        dependency_lock_sha256="c" * 64,
        integration_artifact="artifact",
        runtime_version="revision-v1",
    )


def chunk(chunk_id: str, text: str, *, ordinal: int = 0) -> StoredSupportChunk:
    digest = sha256(text.encode()).hexdigest()
    from app.contracts import DocumentChunk

    return StoredSupportChunk(
        DocumentChunk(
            chunk_id=chunk_id,
            corpus_version_id="corpus-v1",
            page=ordinal + 1,
            section="Section",
            chunk_ordinal=ordinal,
            content_hash=digest,
            text=text,
            embedding=[0.0] * 768,
        ),
        chunk_id,
        "d" * 64,
    )


def hierarchy(lines: tuple[str, ...]) -> tuple[ParentChunk, tuple[ChildDocumentChunk, ...]]:
    text = "\n".join(lines)
    spans: list[SourceSpan] = []
    offset = 0
    for line in lines:
        spans.append(SourceSpan(page=1, start_offset=offset, end_offset=offset + len(line), text=line))
        offset += len(line) + 1
    parent = ParentChunk(
        parent_id="corpus-v1:parent:0",
        corpus_version_id="corpus-v1",
        section_path=("Section",),
        page_start=1,
        page_end=1,
        source_spans=spans,
        content_hash=sha256(text.encode()).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )
    children = tuple(
        ChildDocumentChunk(
            chunk_id=f"corpus-v1:child:{ordinal}",
            corpus_version_id="corpus-v1",
            parent_id=parent.parent_id,
            section_path=parent.section_path,
            page_start=1,
            page_end=1,
            source_spans=[span],
            child_ordinal=ordinal,
            content_hash=sha256(span.text.encode()).hexdigest(),
            token_count=_token_count(span.text),
            text=span.text,
        )
        for ordinal, span in enumerate(spans)
    )
    return parent, children


@dataclass
class Source:
    records: tuple[StoredSupportChunk, ...]
    provenance: str = "d" * 64

    def chunks(self, *, corpus_version_id: str) -> tuple[StoredSupportChunk, ...]:
        assert corpus_version_id == "corpus-v1"
        return self.records

    def source_provenance_sha256(self, *, corpus_version_id: str) -> str:
        assert corpus_version_id == "corpus-v1"
        return self.provenance

    def chunk(self, *, corpus_version_id: str, chunk_id: str) -> StoredSupportChunk | None:
        return next((record for record in self.records if record.chunk_id == chunk_id), None)


class _Snapshot:
    def __init__(self, record_id: str, payload: dict[str, object]) -> None:
        self.id = record_id
        self._payload = payload

    def to_dict(self) -> dict[str, object]:
        return self._payload


class _Collection:
    def __init__(self, records: list[_Snapshot]) -> None:
        self.records = records

    def where(self, **_: object) -> "_Collection":
        return self

    def stream(self) -> list[_Snapshot]:
        return self.records


class _FirestoreClient:
    def __init__(self, collections: dict[str, list[_Snapshot]]) -> None:
        self.collections = collections

    def collection(self, collection_id: str) -> _Collection:
        return _Collection(self.collections.get(collection_id, []))


def run(item_ids: list[str]) -> EvaluationRun:
    return EvaluationRun(
        run_id="run-v1",
        dataset_version_id="dataset-v1",
        corpus_version_id="corpus-v1",
        status="queued",
        selected_item_ids=item_ids,
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        candidate_snapshot=snapshot(),
    )


def test_unique_normalized_match_is_attestation_eligible_and_retry_is_idempotent() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1",
            expected_answer_or_rubric="A",
            review_status="approved",
            supporting_passage="the   exact passage",
        )],
    )
    store = InMemorySupportEvidenceStore()
    source = Source((chunk("chunk-1", "Before THE exact\npassage after"),))
    first = prepare_support(run(["item-1"]), dataset, source, store)
    second = prepare_support(run(["item-1"]), dataset, source, store)

    assert first.evidence[0].match_status == "eligible"
    assert first.evidence[0].selected_chunk is not None
    assert first.marker.approved_item_count == 0
    assert first.marker.eligible_item_count == 1
    assert first.marker == second.marker
    assert first.evidence == second.evidence


def test_hierarchical_support_maps_a_unique_parent_span_to_every_intersecting_child() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    annotation = GoldAnnotation(
        item_id="item-1",
        expected_answer_or_rubric="A",
        review_status="candidate",
        supporting_passage="three four",
    )
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[annotation],
    )
    binding = CandidateGoldBinding(
        corpus_version_id="corpus-v1",
        candidate_snapshot_sha256="e" * 64,
        source_provenance_sha256="d" * 64,
    )
    remap = remap_gold_annotations(dataset, (parent, *children), binding).remaps[0]
    source = Source(tuple(StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent) for child in children))

    evidence = prepare_support(run(["item-1"]), dataset, source, InMemorySupportEvidenceStore()).evidence[0]
    batch_review = create_batch_review(run(["item-1"]), dataset, [evidence])

    child_ids = [child.chunk_id for child in children]
    assert remap.remap_status == "matched"
    assert list(remap.relevant_chunk_ids) == child_ids
    assert evidence.match_status == "eligible"
    assert evidence.candidate_chunk_ids == child_ids
    assert [choice.chunk_id for choice in evidence.selected_chunks] == child_ids
    assert evidence.parent_id == parent.parent_id
    assert evidence.parent_content_hash == parent.content_hash
    assert evidence.parent_span_sha256 == remap.parent_span_sha256
    assert evidence.parent_span_sha256 is not None
    assert batch_review.summary.eligible_count == 1
    assert batch_review.samples[0].parent_span_sha256 == evidence.parent_span_sha256
    assert type(evidence).model_validate(evidence.model_dump()).parent_span_sha256 == evidence.parent_span_sha256
    with pytest.raises(ValueError, match="unique parent source span digest"):
        type(evidence).model_validate(evidence.model_dump() | {"parent_span_sha256": None})


def test_hierarchical_support_repeated_and_absent_parent_spans_remain_unresolved() -> None:
    parent, children = hierarchy(("one repeated support, then repeated support",))
    repeated = GoldAnnotation(
        item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
        supporting_passage="repeated support",
    )
    absent = repeated.model_copy(update={"item_id": "item-2", "supporting_passage": "not present"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id=f"item-{i}", question="Q", dataset_version_id="dataset-v1") for i in (1, 2)],
        gold_annotations=[repeated, absent],
    )
    source = Source((StoredSupportChunk(children[0], children[0].chunk_id, "d" * 64, parent=parent),))

    result = prepare_support(run(["item-1", "item-2"]), dataset, source, InMemorySupportEvidenceStore())
    by_id = {item.item_id: item for item in result.evidence}

    assert by_id["item-1"].match_status == "ambiguous"
    assert by_id["item-1"].selected_chunks == []
    assert by_id["item-2"].match_status == "missing"
    assert by_id["item-2"].selected_chunks == []


def test_hierarchical_manual_review_can_bind_multiple_sibling_children() -> None:
    parent, children = hierarchy(("match here", "match here"))
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="match here",
        )],
    )
    source = Source(tuple(StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent) for child in children))
    run_record = run(["item-1"])
    store = InMemorySupportEvidenceStore()
    prepared = prepare_support(run_record, dataset, source, store)
    selected = tuple(source.chunk(corpus_version_id="corpus-v1", chunk_id=child.chunk_id) for child in children)
    assert all(record is not None for record in selected)

    approved = review_support(
        prepared.evidence[0],
        candidates=tuple(record for record in selected if record is not None),
        reviewer_actor="admin",
        reviewed_at=datetime.now(timezone.utc),
        store=store,
    )
    repeated = review_support(
        approved,
        candidates=tuple(record for record in selected if record is not None),
        reviewer_actor="another-admin",
        reviewed_at=datetime.now(timezone.utc),
        store=store,
    )

    assert prepared.evidence[0].match_status == "ambiguous"
    assert approved.binding_version == "run-bound-support-v2"
    assert [choice.chunk_id for choice in approved.selected_chunks] == [child.chunk_id for child in children]
    from app.evaluation.support import approved_support_chunk_ids

    assert approved_support_chunk_ids(approved, run=run_record) == frozenset(child.chunk_id for child in children)
    assert approved_support_chunk_ids(approved, run=run_record.model_copy(update={"run_id": "stale-run"})) == frozenset()
    assert repeated == approved


def test_hierarchical_support_handles_full_and_subset_runs() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    annotations = [
        GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="three four",
        ),
        GoldAnnotation(
            item_id="item-2", expected_answer_or_rubric="B", review_status="candidate",
            supporting_passage="one two",
        ),
    ]
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[
            EvaluationItem(item_id=f"item-{index}", question="Q", dataset_version_id="dataset-v1")
            for index in (1, 2)
        ],
        gold_annotations=annotations,
    )
    source = Source(tuple(StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent) for child in children))

    subset = prepare_support(
        run(["item-1"]).model_copy(update={"run_id": "subset-run"}),
        dataset,
        source,
        InMemorySupportEvidenceStore(),
    )
    full = prepare_support(
        run(["item-1", "item-2"]).model_copy(update={"run_id": "full-run"}),
        dataset,
        source,
        InMemorySupportEvidenceStore(),
    )

    assert [item.item_id for item in subset.evidence] == ["item-1"]
    assert subset.marker.eligible_item_count == 1
    assert [item.item_id for item in full.evidence] == ["item-1", "item-2"]
    assert full.marker.eligible_item_count == 2


def test_hierarchical_support_rejects_child_with_missing_or_wrong_parent() -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    wrong_parent = children[0].model_copy(update={"parent_id": "corpus-v1:parent:missing"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate",
            supporting_passage="three four",
        )],
    )
    source = Source((
        StoredSupportChunk(wrong_parent, wrong_parent.chunk_id, "d" * 64, parent=parent),
        StoredSupportChunk(children[1], children[1].chunk_id, "d" * 64, parent=parent),
    ))

    evidence = prepare_support(run(["item-1"]), dataset, source, InMemorySupportEvidenceStore()).evidence[0]

    assert evidence.match_status == "invalid"
    assert evidence.reason == "invalid_parent_child_hierarchy"
    assert evidence.selected_chunks == []


@pytest.mark.parametrize("changed", ["page", "offset"])
def test_firestore_support_rejects_changed_hierarchy_provenance_with_unchanged_text_hashes(changed: str) -> None:
    parent, children = hierarchy(("one two three", "four five six"))
    pinned_digest = hierarchical_provenance_sha256((parent,), children)
    if changed == "page":
        parent = parent.model_copy(update={
            "page_start": 2,
            "page_end": 2,
            "source_spans": [span.model_copy(update={"page": 2}) for span in parent.source_spans],
        })
        children = tuple(child.model_copy(update={
            "page_start": 2,
            "page_end": 2,
            "source_spans": [span.model_copy(update={"page": 2}) for span in child.source_spans],
        }) for child in children)
    else:
        def shift(span: SourceSpan) -> SourceSpan:
            return span.model_copy(update={
                "start_offset": span.start_offset + 10,
                "end_offset": span.end_offset + 10,
            })

        parent = parent.model_copy(update={"source_spans": [shift(span) for span in parent.source_spans]})
        children = tuple(child.model_copy(update={"source_spans": [shift(span) for span in child.source_spans]}) for child in children)

    validate_hierarchical_records("corpus-v1", (parent,), children)
    parent_payload = parent.model_dump(mode="json") | {
        "metadata": {"corpus_manifest_sha256": "a" * 64, "derivation_sha256": "b" * 64}
    }
    parent_snapshots = [_Snapshot(parent.parent_id, parent_payload)]
    child_snapshots = []
    for child in children:
        metadata = child.model_dump(mode="json")
        text = metadata.pop("text")
        metadata.update({"corpus_manifest_sha256": "a" * 64, "derivation_sha256": "b" * 64})
        child_snapshots.append(_Snapshot(child.chunk_id, {"text": text, "metadata": metadata}))

    source = FirestoreRunPinnedChunkSource(
        _FirestoreClient({PARENT_COLLECTION: parent_snapshots, "child_vectors": child_snapshots}),
        vector_collection="child_vectors",
        provenance_by_corpus={"corpus-v1": pinned_digest},
        ingestion_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )

    records = source.chunks(corpus_version_id="corpus-v1")

    assert source.hierarchy_error(corpus_version_id="corpus-v1") == (
        "hierarchical source provenance differs from the pinned corpus"
    )
    assert records and all(not record.valid for record in records)
    assert all(
        record.chunk is not None
        and record.chunk.content_hash == sha256(record.chunk.text.encode()).hexdigest()
        for record in records
    )


def test_hierarchical_support_over_cap_is_invalid_without_truncating_child_ids() -> None:
    lines = ("x",) * 65
    parent, children = hierarchy(lines)
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="candidate", supporting_passage="\n".join(lines),
        )],
    )
    source = Source(tuple(StoredSupportChunk(child, child.chunk_id, "d" * 64, parent=parent) for child in children))

    evidence = prepare_support(run(["item-1"]), dataset, source, InMemorySupportEvidenceStore()).evidence[0]

    assert evidence.match_status == "invalid"
    assert evidence.reason == "support_spans_too_many_children"
    assert evidence.candidate_chunk_ids == []


def test_duplicate_missing_boundary_and_unanswerable_items_remain_unresolved() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id=f"item-{i}", question="Q", dataset_version_id="dataset-v1") for i in range(4)],
        gold_annotations=[
            GoldAnnotation(
                item_id="item-0", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="duplicate",
            ),
            GoldAnnotation(
                item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="absent",
            ),
            GoldAnnotation(
                item_id="item-2", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="left right",
            ),
            GoldAnnotation(
                item_id="item-3", expected_answer_or_rubric="A", review_status="approved",
                supporting_passage="ignored",
                semantic_assessment={
                    "response_sha256": "e" * 64, "scorer_version": "v1", "expected_refusal": True,
                },
            ),
        ],
    )
    result = prepare_support(
        run([f"item-{i}" for i in range(4)]),
        dataset,
        Source((
            chunk("a", "duplicate"), chunk("b", "duplicate", ordinal=1),
            chunk("c", "left", ordinal=2), chunk("d", "right", ordinal=3),
        )),
        InMemorySupportEvidenceStore(),
    )
    by_id = {item.item_id: item for item in result.evidence}

    assert by_id["item-0"].match_status == "ambiguous"
    assert by_id["item-0"].candidate_chunk_ids == ["a", "b"]
    assert by_id["item-1"].reason == "missing_supporting_passage"
    assert by_id["item-2"].reason == "support_spans_chunk_boundary"
    assert by_id["item-2"].candidate_chunk_ids == ["c", "d"]
    assert by_id["item-3"].reason == "explicit_unanswerable"
    assert by_id["item-3"].expected_refusal is True
    assert result.marker.approved_item_count == 0


def test_invalid_chunk_metadata_is_unresolved_and_interrupted_retry_has_no_marker() -> None:
    invalid = chunk("bad", "support")
    assert invalid.chunk is not None
    invalid = StoredSupportChunk(
        invalid.chunk.model_copy(update={"content_hash": "f" * 64}),
        "bad",
        "d" * 64,
    )
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            relevant_chunk_ids=["bad"], supporting_passage="support",
        )],
    )
    result = prepare_support(run(["item-1"]), dataset, Source((invalid,)), InMemorySupportEvidenceStore())
    assert result.evidence[0].match_status == "invalid"
    assert result.evidence[0].reason == "invalid_chunk_metadata"

    class FailingStore(InMemorySupportEvidenceStore):
        def create_evidence(self, evidence):  # type: ignore[no-untyped-def]
            raise SupportPreparationError("simulated interruption")

    store = FailingStore()
    with pytest.raises(SupportPreparationError, match="interruption"):
        prepare_support(run(["item-1"]), dataset, Source((chunk("good", "support"),)), store)
    assert store.get_marker(run_id="run-v1") is None


def test_worker_prepares_support_before_scoring() -> None:
    run_record = run(["item-1"])
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(run_record)
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
    )
    events: list[str] = []

    result = EvaluationWorker(runs).execute(
        run_record.run_id,
        dataset,
        lambda _: (events.append("score"), EvaluationOutcome(True, True, True, True, True, 1))[1],
        support_preparer=lambda _run, _dataset: events.append("prepare"),
    )

    assert result.status == "completed"
    assert events == ["prepare", "score"]


def test_stale_run_snapshot_is_rejected_before_any_evidence_is_written() -> None:
    stale = run(["item-1"]).model_copy(update={"corpus_version_id": "other-corpus"})
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
    )
    with pytest.raises(SupportPreparationError, match="not bound"):
        prepare_support(stale, dataset, Source(()), InMemorySupportEvidenceStore())


@pytest.mark.parametrize(
    ("record", "annotation_ids", "expected_reason"),
    [
        (chunk("imported-passage-1", "support"), ["imported-passage-1"], "placeholder_chunk_id"),
        (chunk("real", "support"), ["missing"], "missing_chunk"),
        (StoredSupportChunk(chunk("real", "support").chunk, "other-id", "d" * 64), ["real"], "missing_chunk"),
        (StoredSupportChunk(chunk("real", "support").chunk, "real", "e" * 64), ["real"], "invalid_chunk_metadata"),
        (StoredSupportChunk(chunk("real", "support").chunk.model_copy(update={"content_hash": "f" * 64}), "real", "d" * 64), ["real"], "invalid_chunk_metadata"),
    ],
)
def test_candidate_binding_fails_closed_for_untrusted_chunk_identity(
    record: StoredSupportChunk,
    annotation_ids: list[str],
    expected_reason: str,
) -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            relevant_chunk_ids=annotation_ids, supporting_passage="support",
        )],
    )
    result = prepare_support(run(["item-1"]), dataset, Source((record,)), InMemorySupportEvidenceStore())
    evidence = result.evidence[0]
    assert evidence.match_status in {"missing", "invalid"}
    assert evidence.reason == expected_reason
    assert evidence.selected_chunk is None


def test_eligible_support_never_counts_as_approved_for_retrieval() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            supporting_passage="support",
        )],
    )
    result = prepare_support(run(["item-1"]), dataset, Source((chunk("real", "support"),)), InMemorySupportEvidenceStore())
    assert result.evidence[0].match_status == "eligible"
    from app.evaluation.support import approved_support_chunk_ids

    assert approved_support_chunk_ids(result.evidence[0], run=run(["item-1"])) == frozenset()


def test_exception_review_accepts_a_listed_legacy_chunk_and_is_idempotent() -> None:
    dataset = EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Q", dataset_version_id="dataset-v1")],
        gold_annotations=[GoldAnnotation(
            item_id="item-1", expected_answer_or_rubric="A", review_status="approved",
            supporting_passage="duplicate",
        )],
    )
    store = InMemorySupportEvidenceStore()
    source = Source((chunk("a", "duplicate"), chunk("b", "duplicate", ordinal=1)))
    prepared = prepare_support(run(["item-1"]), dataset, source, store)

    # The candidate must be listed by preparation; a chunk merely existing in
    # the corpus cannot be smuggled into a review decision.
    with pytest.raises(SupportPreparationError, match="server-listed"):
        review_support(
            prepared.evidence[0], candidates=(chunk("unlisted", "support"),),
            reviewer_actor="admin", reviewed_at=datetime.now(timezone.utc), store=store,
        )

    selected = source.chunk(corpus_version_id="corpus-v1", chunk_id="a")
    assert selected is not None
    approved = review_support(
        prepared.evidence[0], candidates=(selected,),
        reviewer_actor="admin", reviewed_at=datetime.now(timezone.utc), store=store,
    )
    repeated = review_support(
        approved, candidates=(selected,),
        reviewer_actor="another-admin", reviewed_at=datetime.now(timezone.utc), store=store,
    )
    assert approved.match_status == "review_approved"
    assert approved.selected_chunk is not None and approved.selected_chunk.chunk_id == "a"
    assert repeated == approved
