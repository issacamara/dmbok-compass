from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import pytest

from app.contracts import (
    AllowanceReservation,
    ChildDocumentChunk,
    CorpusVersion,
    DatasetHead,
    EvaluationRowV2,
    ParentChunk,
    RawSourceSpan,
    SourceSpan,
)
from app.corpus.chunking import _token_count
from app.corpus.versions import hierarchical_validation_evidence
from app.evaluation.candidate_generation import (
    CandidateAllowanceReconciliationError,
    CandidateSourceError,
    _validate_sources,
    generate_candidate_revision,
)
from app.provider_allowance import (
    AllowanceStageResult,
    InMemoryProviderAllowanceStore,
    ProviderAccountCapabilities,
    ProviderAllowanceObservation,
    provider_allowance_reservation_id,
)
from model.candidate_generation import CandidateDraft, CandidateGenerationResult
from model.protocol import (
    GenerationError,
    GenerationErrorCode,
    UsageMetadata,
)


NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)
CORPUS = "corpus-v2"
QUOTE = "Data quality requires clear ownership."


def source_records():
    span = SourceSpan(
        page=21,
        start_offset=0,
        end_offset=len(QUOTE),
        text=QUOTE,
        span_id="span-21-0",
        block_id="block-21-0",
    )
    parent = ParentChunk(
        parent_id="corpus-v2:parent:1:1",
        corpus_version_id=CORPUS,
        section_path=("Data Quality",),
        page_start=21,
        page_end=21,
        source_spans=[span],
        source_text=QUOTE,
        content_hash=hashlib.sha256(("Data Quality\n" + QUOTE).encode()).hexdigest(),
        token_count=_token_count("Data Quality\n" + QUOTE),
        text="Data Quality\n" + QUOTE,
    )
    child = ChildDocumentChunk(
        chunk_id="corpus-v2:child:21-0",
        corpus_version_id=CORPUS,
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=21,
        page_end=21,
        source_spans=[span],
        source_text=QUOTE,
        child_ordinal=0,
        content_hash=hashlib.sha256(parent.text.encode()).hexdigest(),
        token_count=_token_count(parent.text),
        text=parent.text,
    )
    return (child,), (parent,)


def answer_candidate(question: str, *, category: str = "definitions", outcome: str = "answer"):
    value = {
        "question": question,
        "category": category,
        "expected_outcome": outcome,
        "expected_answer_or_rubric": "Explain the relevant DMBOK concept.",
        "supports": [{"source_id": "source-0", "quote": QUOTE}],
        "scope_rationale": None,
    }
    return value


def refusal_candidate(question: str, *, category: str = "scenarios"):
    return {
        "question": question,
        "category": category,
        "expected_outcome": "evidence_refusal",
        "expected_answer_or_rubric": "Refuse and explain that this is outside the corpus scope.",
        "supports": [],
        "scope_rationale": "The question asks for information outside the supplied DMBOK corpus.",
    }


def result(candidates, *, provider="openrouter", model="generator-v1", usage=12):
    return CandidateGenerationResult(
        candidates=tuple(CandidateDraft.model_validate(candidate) for candidate in candidates),
        usage=UsageMetadata(provider=provider, model=model, total_tokens=usage),
    )


class FakeCorpusStore:
    def __init__(self, version):
        self.version = version
        self.reads = 0

    def get(self, version_id):
        self.reads += 1
        if self.version is not None and self.version.version_id != version_id:
            return None
        return self.version


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.max_output_tokens = 512

    async def generate(self, request):
        self.calls += 1
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


class FakeStore:
    def __init__(self, rows=()):
        self.rows = tuple(rows)
        self.head = None
        self.staged = 0
        self.head_reads = 0
        self.revision_reads = 0

    def corpus_head(self, corpus_version_id):
        self.head_reads += 1
        return self.head

    def get_revision(self, head):
        self.revision_reads += 1
        assert head == self.head
        return self.rows

    def stage_revision(self, corpus_version_id, revision, rows):
        self.staged += 1
        head = DatasetHead(
            corpus_version_id=corpus_version_id,
            revision=revision,
            dataset_sha256=hashlib.sha256(
                "|".join(row.item_id for row in rows).encode()
            ).hexdigest(),
        )
        self._staged_rows = tuple(rows)
        return head

    def activate_revision(self, head, *, expected_revision):
        assert expected_revision == (self.head.revision if self.head else None)
        self.head = head
        self.rows = self._staged_rows
        return head


class FakeAllowanceStore:
    def __init__(self, reservation, *, forced_status=None):
        self.reservation = reservation
        self.reconciliations = []
        self.forced_status = forced_status

    def reconcile_usage(self, capabilities, operation_key, actual_units):
        self.reconciliations.append((capabilities, operation_key, actual_units))
        status = self.forced_status or ("staged" if actual_units is not None else "awaiting_budget")
        return AllowanceStageResult(status, self.reservation)

def allowance(status="staged", *, kind="generation", units=1_000_000):
    capabilities = ProviderAccountCapabilities("openrouter", "generator", kind)
    operation_key = "generation-1"
    reservation = None if status != "staged" else AllowanceReservation(
        reservation_id=provider_allowance_reservation_id(capabilities, operation_key),
        operation_key=operation_key,
        kind=kind,
        allowance_observed_at=NOW - timedelta(minutes=1),
        allowance_expires_at=NOW + timedelta(minutes=10),
        reserved_units=units,
    )
    return AllowanceStageResult(status=status, reservation=reservation, reason="refresh account allowance")


def reserved_generation(estimated_units):
    capabilities = ProviderAccountCapabilities(
        "openrouter", "generator", "generation", max_units_per_operation=2_000_000, max_retries=1
    )
    store = InMemoryProviderAllowanceStore()
    store.observe(
        capabilities,
        ProviderAllowanceObservation(
            4_000_000,
            NOW - timedelta(minutes=1),
            NOW + timedelta(minutes=10),
        ),
    )
    budget = store.stage_reservation(
        capabilities, "generation-1", estimated_units, retries=1, now=NOW
    )
    return store, budget, capabilities


def existing_row():
    child = source_records()[0][0]
    span = child.source_spans[0]
    raw_span = RawSourceSpan(
        span_id=span.span_id,
        block_id=span.block_id,
        physical_page=span.page,
        start_offset=span.start_offset,
        end_offset=span.end_offset,
        raw_text=span.text,
    )
    return EvaluationRowV2(
        item_id="old-row", origin="admin", question="Existing row?", category="definitions",
        expected_outcome="answer", expected_answer_or_rubric="Keep this row", expected_refusal=False,
        source_spans=(raw_span,), relevant_child_ids=(child.chunk_id,), parent_ids=(child.parent_id,),
    )


def ready_corpus_store(children, parents, *, status="validated", evidence_update=None):
    version = CorpusVersion(
        version_id=CORPUS,
        source_uri="gs://corpus/dmbok.pdf",
        embedding_model="text-embedding-005",
        status=status,
        created_at=NOW,
    )
    evidence = hierarchical_validation_evidence(
        version, parents, children, index_ready=True, filter_ready=True
    )
    if evidence_update:
        evidence = evidence.model_copy(update=evidence_update)
    return FakeCorpusStore(version.model_copy(update={"validation_evidence": evidence}))


def generate(
    adapter,
    store,
    *,
    value=None,
    budget=None,
    target_count=8,
    children=None,
    parents=None,
    corpus_store=None,
    allowance_store=None,
    allowance_capabilities=None,
    operation_key="generation-1",
):
    if children is None or parents is None:
        children, parents = source_records()
    corpus_store = corpus_store or ready_corpus_store(children, parents)
    budget = budget or allowance()
    allowance_store = allowance_store or FakeAllowanceStore(budget.reservation)
    allowance_capabilities = allowance_capabilities or ProviderAccountCapabilities(
        "openrouter", "generator", "generation"
    )
    return asyncio.run(generate_candidate_revision(
        corpus_version_id=CORPUS,
        children=children,
        parents=parents,
        corpus_store=corpus_store,
        adapter=adapter,
        generator_route="openrouter",
        generator_model="generator-v1",
        generator_configuration_version="generator-config-v1",
        operation_key=operation_key,
        allowance=budget,
        allowance_store=allowance_store,
        allowance_capabilities=allowance_capabilities,
        store=store,
        target_count=target_count,
        now=NOW,
    ))


def test_generation_covers_categories_outcomes_and_reports_usage_and_gaps():
    drafts = [
        answer_candidate("What is data quality?", category="definitions"),
        answer_candidate("Why validate data?", category="explanations", outcome="qualified"),
        answer_candidate("How do quality controls compare?", category="comparisons"),
        answer_candidate("How should a steward study quality?", category="study"),
        refusal_candidate("What is the weather in Rabat?", category="scenarios"),
    ]
    store = FakeStore()
    report = generate(FakeAdapter([result(drafts)]), store, target_count=5)

    assert report.status == "published"
    assert report.generated_count == 5
    assert report.achievable_size == 5
    assert report.coverage_by_category == {
        "definitions": 1, "explanations": 1, "comparisons": 1, "study": 1, "scenarios": 1,
    }
    assert report.coverage_by_outcome == {"answer": 3, "qualified": 1, "evidence_refusal": 1}
    assert report.gaps == ()
    assert report.usage_tokens == 12
    assert report.generator_configuration_version == "generator-config-v1"
    assert report.dataset_head == store.head


def test_generation_reconciles_known_usage_and_holds_unknown_usage():
    known_allowances, budget, capabilities = reserved_generation(500_000)
    known_adapter = FakeAdapter([result([answer_candidate("Define data quality.")], usage=17)])
    known_report = generate(
        known_adapter,
        FakeStore(),
        budget=budget,
        allowance_store=known_allowances,
        allowance_capabilities=capabilities,
        target_count=1,
    )
    assert known_report.status == "published"
    account_key = (capabilities.provider_id, capabilities.account_id, capabilities.kind)
    assert known_allowances.spent_units[account_key] == 17
    assert known_allowances.reserved_units[account_key] == 0

    unknown_allowances, unknown_budget, unknown_capabilities = reserved_generation(500_000)
    unknown_adapter = FakeAdapter([
        GenerationError(GenerationErrorCode.UNAVAILABLE, "uncertain", retryable=False)
    ])
    unknown_report = generate(
        unknown_adapter,
        FakeStore(),
        budget=unknown_budget,
        allowance_store=unknown_allowances,
        allowance_capabilities=unknown_capabilities,
        target_count=1,
    )
    assert unknown_report.status == "failed"
    unknown_key = (
        unknown_capabilities.provider_id,
        unknown_capabilities.account_id,
        unknown_capabilities.kind,
    )
    assert unknown_allowances.reserved_units[unknown_key] == unknown_budget.reservation.reserved_units
    assert unknown_allowances.stage_reservation(
        unknown_capabilities, "generation-1", 1, now=NOW
    ).status == "awaiting_budget"


def test_insufficient_generation_reservation_reconciles_zero_without_calls():
    allowances, budget, capabilities = reserved_generation(1)
    adapter = FakeAdapter([result([answer_candidate("Must not run")])])
    store = FakeStore()

    report = generate(
        adapter,
        store,
        budget=budget,
        allowance_store=allowances,
        allowance_capabilities=capabilities,
        target_count=1,
    )

    assert report.status == "awaiting_budget"
    assert "bounded operation estimate" in (report.error or "")
    assert adapter.calls == 0
    assert store.head_reads == 0
    assert store.revision_reads == 0
    assert store.staged == 0
    account_key = (capabilities.provider_id, capabilities.account_id, capabilities.kind)
    assert allowances.reserved_units[account_key] == 0
    assert allowances.spent_units[account_key] == 0


def test_generation_rejects_wrong_provider_or_operation_reservation_before_calls():
    for capabilities, operation_key, error in (
        (
            ProviderAccountCapabilities("google-gemini", "generator", "generation"),
            "generation-1",
            "does not match",
        ),
        (
            ProviderAccountCapabilities("openrouter", "generator", "generation"),
            "another-operation",
            "wrong operation key",
        ),
        (
            ProviderAccountCapabilities("openrouter", "other-generator", "generation"),
            "generation-1",
            "provider account identity",
        ),
    ):
        budget = allowance()
        allowances = FakeAllowanceStore(budget.reservation)
        adapter = FakeAdapter([result([answer_candidate("Must not run")])])
        store = FakeStore()
        with pytest.raises(ValueError, match=error):
            generate(
                adapter,
                store,
                budget=budget,
                allowance_store=allowances,
                allowance_capabilities=capabilities,
                operation_key=operation_key,
                target_count=1,
            )
        assert adapter.calls == 0
        assert store.head_reads == 0
        assert store.staged == 0
        assert allowances.reconciliations == []


def test_failed_known_usage_reconciliation_prevents_dataset_publication():
    budget = allowance()
    allowances = FakeAllowanceStore(budget.reservation, forced_status="awaiting_budget")
    adapter = FakeAdapter([result([answer_candidate("Reconcile before publish")], usage=19)])
    store = FakeStore()

    with pytest.raises(CandidateAllowanceReconciliationError, match="not confirmed"):
        generate(
            adapter,
            store,
            budget=budget,
            allowance_store=allowances,
            target_count=1,
        )

    assert adapter.calls == 1
    assert store.staged == 0
    assert store.head is None
    assert [entry[2] for entry in allowances.reconciliations] == [19]


def test_candidates_bind_exact_corpus_spans_and_remain_unapproved():
    store = FakeStore()
    report = generate(FakeAdapter([result([answer_candidate("Define data quality.")])]), store, target_count=1)

    row = store.rows[0]
    assert report.status == "published"
    assert row.origin == "generated"
    assert row.expected_refusal is False
    assert row.source_spans[0].span_id == "span-21-0"
    assert row.source_spans[0].physical_page == 21
    assert row.relevant_child_ids == ("corpus-v2:child:21-0",)
    assert row.parent_ids == ("corpus-v2:parent:1:1",)
    assert not hasattr(row, "review_status")


def test_child_source_binding_rejects_corrupt_page_and_text():
    children, parents = source_records()
    child = children[0]
    original_span = child.source_spans[0]

    wrong_page_span = original_span.model_copy(update={"page": original_span.page + 1})
    wrong_page_child = child.model_copy(
        update={
            "source_spans": [wrong_page_span],
            "page_start": original_span.page + 1,
            "page_end": original_span.page + 1,
        }
    )
    with pytest.raises(CandidateSourceError, match="one parent source span"):
        _validate_sources(CORPUS, (wrong_page_child,), parents)

    wrong_identity_span = original_span.model_copy(update={"span_id": "wrong-span-id"})
    wrong_identity_child = child.model_copy(update={"source_spans": [wrong_identity_span]})
    with pytest.raises(CandidateSourceError, match="span identity"):
        _validate_sources(CORPUS, (wrong_identity_child,), parents)

    wrong_text_span = original_span.model_copy(update={"text": "Corrupted source excerpt."})
    wrong_text_child = child.model_copy(
        update={"source_spans": [wrong_text_span], "source_text": wrong_text_span.text}
    )
    with pytest.raises(CandidateSourceError, match="does not match its parent source range"):
        _validate_sources(CORPUS, (wrong_text_child,), parents)


def test_generation_requires_ready_complete_corpus_before_model_or_dataset_access():
    children, parents = source_records()
    version_without_evidence = CorpusVersion(
        version_id=CORPUS,
        source_uri="gs://corpus/dmbok.pdf",
        embedding_model="text-embedding-005",
        status="validated",
        created_at=NOW,
    )
    incomplete_stores = (
        FakeCorpusStore(None),
        ready_corpus_store(children, parents, status="staged"),
        ready_corpus_store(children, parents, evidence_update={"filter_ready": False}),
        FakeCorpusStore(version_without_evidence),
    )
    for corpus_store in incomplete_stores:
        adapter = FakeAdapter([result([answer_candidate("Must not run")])])
        revision_store = FakeStore()
        with pytest.raises(CandidateSourceError):
            generate(
                adapter,
                revision_store,
                children=children,
                parents=parents,
                corpus_store=corpus_store,
                target_count=1,
            )
        assert adapter.calls == 0
        assert revision_store.head_reads == 0
        assert revision_store.revision_reads == 0
        assert revision_store.staged == 0

    bad_hash_child = children[0].model_copy(update={"content_hash": "0" * 64})
    bad_hash_children = (bad_hash_child,)
    adapter = FakeAdapter([result([answer_candidate("Must not run with a bad hash")])])
    revision_store = FakeStore()
    with pytest.raises(CandidateSourceError, match="completeness validation"):
        generate(
            adapter,
            revision_store,
            children=bad_hash_children,
            parents=parents,
            corpus_store=ready_corpus_store(bad_hash_children, parents),
            target_count=1,
        )
    assert adapter.calls == 0
    assert revision_store.head_reads == 0
    assert revision_store.staged == 0


def test_generation_rejects_near_duplicate_questions():
    drafts = [
        answer_candidate(
            "How are data governance stewardship responsibilities and accountability roles applied across enterprise policy management?"
        ),
        answer_candidate(
            "What are data governance stewardship responsibilities and authority roles applied across enterprise policy management?"
        ),
    ]
    store = FakeStore()
    adapter = FakeAdapter([
        result(drafts),
        result([answer_candidate("Which governance model controls stewardship policy management?")]),
    ])
    report = generate(adapter, store, target_count=2)

    assert report.status == "published"
    assert report.generated_count == 2
    assert report.rejected_count == 1
    assert len(store.rows) == 2


def test_refusal_proposals_require_scope_rationale_and_no_support():
    store = FakeStore()
    report = generate(FakeAdapter([result([refusal_candidate("What is the weather?")])]), store, target_count=1)

    assert report.status == "published"
    refusal = store.rows[0]
    assert refusal.expected_refusal is True
    assert refusal.scope_rationale
    assert refusal.source_spans == ()
    assert refusal.relevant_child_ids == ()
    assert refusal.parent_ids == ()


def test_retryable_failure_retries_inside_allowance_and_keeps_old_rows():
    old = existing_row()
    store = FakeStore([old])
    store.head = DatasetHead(corpus_version_id=CORPUS, revision=1, dataset_sha256="a" * 64)
    adapter = FakeAdapter([
        GenerationError(GenerationErrorCode.UNAVAILABLE, "temporary", retryable=True),
        result([answer_candidate("A new definition?")]),
    ])

    report = generate(adapter, store, target_count=1)

    assert adapter.calls == 2
    assert report.status == "published"
    assert store.head.revision == 2
    assert store.rows[0] == old
    assert len(store.rows) == 2


def test_provider_failure_invalid_output_duplicates_and_budget_pause_preserve_old_head():
    old = existing_row()
    for adapter, budget in (
        (FakeAdapter([GenerationError(GenerationErrorCode.UNAVAILABLE, "down", retryable=False)]), allowance()),
        (FakeAdapter([GenerationError(GenerationErrorCode.INVALID_OUTPUT, "invalid")]), allowance()),
        (FakeAdapter([result([answer_candidate("Existing row?")])]), allowance()),
        (FakeAdapter([result([answer_candidate("Must not be called")])]), allowance("awaiting_budget")),
    ):
        store = FakeStore([old])
        store.head = DatasetHead(corpus_version_id=CORPUS, revision=1, dataset_sha256="a" * 64)
        report = generate(adapter, store, budget=budget, target_count=1)
        assert report.status in {"failed", "awaiting_budget"}
        assert store.head.revision == 1
        assert store.rows == (old,)
        assert store.staged == 0
    assert adapter.calls == 0


def test_invalid_child_binding_and_wrong_pinned_model_never_publish():
    store = FakeStore()
    bad_source = answer_candidate("Broken support?")
    bad_source["supports"] = [{"source_id": "unknown", "quote": QUOTE}]
    report = generate(FakeAdapter([result([bad_source])]), store, target_count=1)
    assert report.status == "failed"
    assert store.head is None

    store = FakeStore()
    report = generate(FakeAdapter([result([answer_candidate("Wrong model?")], model="other")]), store, target_count=1)
    assert report.status == "failed"
    assert "pinned model route" in (report.error or "")
    assert store.head is None
