"""Generate a provisional evaluation revision from one ready parsed corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
import unicodedata
from typing import Literal, Protocol

from pydantic import ValidationError

from app.contracts import (
    AllowanceReservation,
    ChildDocumentChunk,
    CorpusVersion,
    DatasetHead,
    EvaluationRowV2,
    ParentChunk,
    RawSourceSpan,
)
from app.evaluation.generations import ConcurrentActivationError
from app.corpus.versions import (
    CorpusCompletenessError,
    hierarchical_validation_evidence,
    validate_hierarchical_records,
)
from app.corpus.chunking import _token_count
from app.provider_allowance import (
    AllowanceStageResult,
    ProviderAccountCapabilities,
    ProviderAllowanceStore,
)
from model.candidate_generation import CandidateDraft, CandidateGenerationAdapter, PROMPT_VERSION
from model.protocol import GenerationError, GenerationRequest, Passage


_CATEGORIES = ("definitions", "explanations", "comparisons", "study", "scenarios")
_OUTCOMES = ("answer", "qualified", "evidence_refusal")
_BATCH_SIZE = 8
_MAX_BATCHES = 50
_MAX_RETRIES = 1
_GENERATION_PROMPT_OVERHEAD_TOKENS = 256
_GENERATION_PASSAGE_OVERHEAD_TOKENS = 32
_TOKEN_ESTIMATE_SAFETY_FACTOR = 2
_MODEL_PROVIDER = {"openrouter": "openrouter", "vertex": "google-gemini"}
_QUESTION_FILLER = frozenset(
    {
        "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "could", "describe",
        "define", "do", "does", "for", "from", "give", "how", "in", "is", "it", "of", "on",
        "or", "please", "should", "that", "the", "their", "this", "to", "what", "when", "where",
        "which", "who", "why", "with", "would",
    }
)
_NEAR_DUPLICATE_THRESHOLD = 0.8
_NEAR_DUPLICATE_MIN_TOKENS = 5


class CandidateRevisionStore(Protocol):
    def corpus_head(self, corpus_version_id: str) -> DatasetHead | None: ...

    def get_revision(self, head: DatasetHead) -> tuple[EvaluationRowV2, ...]: ...

    def stage_revision(
        self, corpus_version_id: str, revision: int, rows: tuple[EvaluationRowV2, ...]
    ) -> DatasetHead: ...

    def activate_revision(self, head: DatasetHead, *, expected_revision: int | None) -> DatasetHead: ...


class CandidateCorpusStore(Protocol):
    def get(self, version_id: str) -> CorpusVersion | None: ...


@dataclass(frozen=True)
class CandidateGenerationReport:
    status: Literal["published", "awaiting_budget", "failed"]
    corpus_version_id: str
    generator_policy_version: str
    generator_configuration_version: str
    generator_model: str
    target_count: int
    generated_count: int
    rejected_count: int
    achievable_size: int
    coverage_by_category: dict[str, int]
    coverage_by_outcome: dict[str, int]
    gaps: tuple[str, ...]
    usage_tokens: int | None
    dataset_head: DatasetHead | None = None
    error: str | None = None


@dataclass
class _UsageAccumulator:
    tokens: int = 0
    known: bool = True
    request_in_flight: bool = False
    reconciled: bool = False


class CandidateSourceError(ValueError):
    """Raised when the supplied ready-corpus source set is not self-consistent."""


class CandidateAllowanceReconciliationError(RuntimeError):
    """Raised when known provider usage cannot be recorded by the allowance ledger."""


async def generate_candidate_revision(
    *,
    corpus_version_id: str,
    children: tuple[ChildDocumentChunk, ...],
    parents: tuple[ParentChunk, ...],
    corpus_store: CandidateCorpusStore,
    adapter: CandidateGenerationAdapter,
    generator_route: str,
    generator_model: str,
    generator_configuration_version: str,
    operation_key: str,
    allowance: AllowanceStageResult,
    allowance_store: ProviderAllowanceStore,
    allowance_capabilities: ProviderAccountCapabilities,
    store: CandidateRevisionStore,
    target_count: int = 200,
    now: datetime | None = None,
) -> CandidateGenerationReport:
    """Generate, validate, then atomically publish one candidate dataset revision."""
    if allowance.status != "staged" or allowance.reservation is None:
        if target_count < 1:
            raise ValueError("target_count must be positive")
        if target_count > 200:
            raise ValueError("target_count cannot exceed the initial 200-row candidate target")
        return _report(
            "awaiting_budget", corpus_version_id, generator_configuration_version,
            generator_model, target_count, (), 0, None, error=allowance.reason,
        )
    reservation = allowance.reservation
    if reservation.kind != "generation" or allowance_capabilities.kind != "generation":
        raise ValueError("candidate generation requires a generation allowance reservation")
    if not operation_key.strip() or reservation.operation_key != operation_key:
        raise ValueError("candidate generation allowance reservation has the wrong operation key")
    if generator_route not in _MODEL_PROVIDER:
        raise ValueError("candidate generation requires a supported generator route")
    if allowance_capabilities.provider_id != _MODEL_PROVIDER[generator_route]:
        raise ValueError("candidate generator route does not match its allowance provider")
    usage = _UsageAccumulator()
    try:
        if target_count < 1:
            raise ValueError("target_count must be positive")
        if target_count > 200:
            raise ValueError("target_count cannot exceed the initial 200-row candidate target")
        if not _reservation_is_current(reservation, now=now):
            usage.known = False
            return _report(
                "awaiting_budget", corpus_version_id, generator_configuration_version,
                generator_model, target_count, (), 0, None, error="generation allowance is unavailable",
            )
        if generator_route not in _MODEL_PROVIDER or not generator_model.strip() or not generator_configuration_version.strip():
            raise ValueError("candidate generation requires a pinned supported route, model, and configuration")
        _validate_ready_corpus(corpus_version_id, corpus_store, children, parents)
        source_map = _validate_sources(corpus_version_id, children, parents)
        if not source_map:
            raise CandidateSourceError("candidate generation requires verified child sources")
        required_units = _required_reservation_units(target_count, tuple(source_map.values()), adapter)
        if reservation.reserved_units < required_units:
            return _report(
                "awaiting_budget", corpus_version_id, generator_configuration_version,
                generator_model, target_count, (), 0, 0,
                error="generation reservation is smaller than the bounded operation estimate",
            )
        return await _generate_candidate_revision(
            corpus_version_id=corpus_version_id,
            source_map=source_map,
            adapter=adapter,
            generator_route=generator_route,
            generator_model=generator_model,
            generator_configuration_version=generator_configuration_version,
            allowance_reservation=reservation,
            allowance_store=allowance_store,
            allowance_capabilities=allowance_capabilities,
            store=store,
            target_count=target_count,
            usage=usage,
        )
    finally:
        if not usage.reconciled:
            _reconcile_usage(
                allowance_store, allowance_capabilities, reservation, usage
            )


def _reconcile_usage(
    allowance_store: ProviderAllowanceStore,
    capabilities: ProviderAccountCapabilities,
    reservation: AllowanceReservation,
    usage: _UsageAccumulator,
) -> None:
    actual_units = None if not usage.known or usage.request_in_flight else usage.tokens
    usage.reconciled = True
    result = allowance_store.reconcile_usage(
        capabilities, reservation.operation_key, actual_units
    )
    expected_status = "staged" if actual_units is not None else "awaiting_budget"
    if result.status != expected_status or result.reservation != reservation:
        raise CandidateAllowanceReconciliationError(
            "candidate generation allowance reconciliation was not confirmed"
        )


def _required_reservation_units(
    target_count: int,
    children: tuple[ChildDocumentChunk, ...],
    adapter: CandidateGenerationAdapter,
) -> int:
    calls = _generation_call_limit(target_count)
    max_output_tokens = getattr(adapter, "max_output_tokens", None)
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("candidate generator must expose a positive max_output_tokens limit")
    prompt_tokens = _token_count(_prompt(target_count, [])) + _GENERATION_PROMPT_OVERHEAD_TOKENS
    max_passage_tokens = 0
    for batch_index in range(calls):
        batch = _select_sources(children, batch_index)
        token_count = sum(
            _token_count((child.source_text or "\n".join(span.text for span in child.source_spans))[:1600])
            + _GENERATION_PASSAGE_OVERHEAD_TOKENS
            for child in batch
        )
        max_passage_tokens = max(max_passage_tokens, token_count)
    input_estimate = (prompt_tokens + max_passage_tokens) * _TOKEN_ESTIMATE_SAFETY_FACTOR
    units_per_attempt = input_estimate + max_output_tokens
    return calls * units_per_attempt * (1 + _MAX_RETRIES)


def _generation_call_limit(target_count: int) -> int:
    return min(_MAX_BATCHES, max(1, (target_count + _BATCH_SIZE - 1) // _BATCH_SIZE * 2))


async def _generate_candidate_revision(
    *,
    corpus_version_id: str,
    source_map: dict[str, ChildDocumentChunk],
    adapter: CandidateGenerationAdapter,
    generator_route: str,
    generator_model: str,
    generator_configuration_version: str,
    allowance_reservation: AllowanceReservation,
    allowance_store: ProviderAllowanceStore,
    allowance_capabilities: ProviderAccountCapabilities,
    store: CandidateRevisionStore,
    target_count: int,
    usage: _UsageAccumulator,
) -> CandidateGenerationReport:

    head = store.corpus_head(corpus_version_id)
    current_rows = store.get_revision(head) if head else ()
    old_head = head
    seen_questions = {_normalize_question_key(row.question) for row in current_rows}
    rows: list[EvaluationRowV2] = []
    rejected = 0
    calls = _generation_call_limit(target_count)

    for batch_index in range(calls):
        if len(rows) >= target_count:
            break
        selections = _select_sources(tuple(source_map.values()), batch_index)
        source_by_alias = {
            f"source-{index}": (
                child,
                (child.source_text or "\n".join(span.text for span in child.source_spans))[:1600],
            )
            for index, child in enumerate(selections)
        }
        passages = [
            Passage(
                passage_id=alias,
                page=child.page_start,
                section=" / ".join(child.section_path),
                text=sent_text,
            )
            for alias, (child, sent_text) in source_by_alias.items()
        ]
        prompt = _prompt(target_count - len(rows), rows)
        result = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                usage.request_in_flight = True
                result = await adapter.generate(GenerationRequest(prompt=prompt, passages=passages))
                usage.request_in_flight = False
                break
            except GenerationError as exc:
                usage.request_in_flight = False
                usage.known = False
                if not exc.retryable or attempt >= _MAX_RETRIES:
                    return _report(
                        "failed", corpus_version_id, generator_configuration_version,
                        generator_model, target_count, tuple(rows), rejected, _usage(usage.tokens, usage.known),
                        error="candidate generation failed",
                    )
            except Exception:
                usage.request_in_flight = False
                usage.known = False
                return _report(
                    "failed", corpus_version_id, generator_configuration_version, generator_model,
                    target_count, tuple(rows), rejected, _usage(usage.tokens, usage.known),
                    error="candidate generation failed",
                )
        assert result is not None
        if result.usage.total_tokens is None:
            usage.known = False
        elif usage.known:
            usage.tokens += result.usage.total_tokens
        if (
            result.usage.provider != _MODEL_PROVIDER[generator_route]
            or result.usage.model != generator_model
        ):
            return _report(
                "failed", corpus_version_id, generator_configuration_version, generator_model,
                target_count, tuple(rows), rejected, _usage(usage.tokens, usage.known),
                error="generator response did not match the pinned model route",
            )
        try:
            drafts = result.candidates
        except (ValidationError, ValueError):
            rejected += 1
            continue
        for draft in drafts:
            if len(rows) >= target_count:
                break
            question_key = _normalize_question_key(draft.question)
            if _is_near_duplicate(draft.question, seen_questions):
                rejected += 1
                continue
            row = _to_row(corpus_version_id, draft, source_by_alias)
            if row is None:
                rejected += 1
                continue
            rows.append(row)
            seen_questions.add(question_key)

    if not rows:
        return _report(
            "failed", corpus_version_id, generator_configuration_version, generator_model,
            target_count, (), rejected, _usage(usage.tokens, usage.known), error="no valid candidates were generated",
        )
    _reconcile_usage(
        allowance_store, allowance_capabilities, allowance_reservation, usage
    )
    if store.corpus_head(corpus_version_id) != old_head:
        raise ConcurrentActivationError("corpus dataset changed during candidate generation")
    next_revision = (old_head.revision + 1) if old_head else 1
    combined_rows = (*current_rows, *rows)
    staged = store.stage_revision(corpus_version_id, next_revision, combined_rows)
    activated = store.activate_revision(staged, expected_revision=old_head.revision if old_head else None)
    return _report(
        "published", corpus_version_id, generator_configuration_version, generator_model,
        target_count, tuple(rows), rejected, _usage(usage.tokens, usage.known), dataset_head=activated,
    )


def _reservation_is_current(reservation: AllowanceReservation, *, now: datetime | None) -> bool:
    instant = now or datetime.now(timezone.utc)
    return reservation.allowance_observed_at <= instant < reservation.allowance_expires_at


def _validate_ready_corpus(
    corpus_version_id: str,
    corpus_store: CandidateCorpusStore,
    children: tuple[ChildDocumentChunk, ...],
    parents: tuple[ParentChunk, ...],
) -> CorpusVersion:
    version = corpus_store.get(corpus_version_id)
    if version is None or version.version_id != corpus_version_id:
        raise CandidateSourceError("candidate generation requires a selected corpus version")
    evidence = version.validation_evidence
    if (
        version.status not in {"validated", "active"}
        or evidence is None
        or not evidence.index_ready
        or not evidence.filter_ready
        or evidence.chunk_count != len(children)
    ):
        raise CandidateSourceError("candidate generation requires a complete ready corpus")
    try:
        validate_hierarchical_records(corpus_version_id, parents, children)
        expected = hierarchical_validation_evidence(
            version, parents, children, index_ready=True, filter_ready=True
        )
    except (CorpusCompletenessError, ValueError) as exc:
        raise CandidateSourceError("candidate corpus completeness validation failed") from exc
    if evidence != expected:
        raise CandidateSourceError("candidate corpus readiness evidence does not match its source records")
    return version


def _validate_sources(
    corpus_version_id: str,
    children: tuple[ChildDocumentChunk, ...],
    parents: tuple[ParentChunk, ...],
) -> dict[str, ChildDocumentChunk]:
    parent_map = {parent.parent_id: parent for parent in parents}
    child_map = {child.chunk_id: child for child in children}
    if len(parent_map) != len(parents) or len(child_map) != len(children):
        raise CandidateSourceError("candidate source identities must be unique")
    for parent in parents:
        if parent.corpus_version_id != corpus_version_id:
            raise CandidateSourceError("candidate parents must belong to the requested corpus")
    for child in children:
        parent = parent_map.get(child.parent_id)
        if child.corpus_version_id != corpus_version_id or parent is None:
            raise CandidateSourceError("candidate children must link to a parent in the requested corpus")
        if parent.corpus_version_id != child.corpus_version_id:
            raise CandidateSourceError("candidate child and parent corpus versions must match")
        for span in child.source_spans:
            matching_parent_spans = [
                parent_span
                for parent_span in parent.source_spans
                if parent_span.block_id is not None
                and parent_span.block_id == span.block_id
                and parent_span.page == span.page
                and parent_span.start_offset <= span.start_offset
                and span.end_offset <= parent_span.end_offset
            ]
            if len(matching_parent_spans) != 1:
                raise CandidateSourceError("candidate child spans must identify one parent source span")
            parent_span = matching_parent_spans[0]
            if parent_span.span_id is None or span.span_id is None:
                raise CandidateSourceError("candidate child and parent spans require source identities")
            if span.start_offset == parent_span.start_offset and span.end_offset == parent_span.end_offset:
                expected_span_id = parent_span.span_id
            else:
                expected_span_id = hashlib.sha256(
                    f"{span.block_id}:{span.start_offset}:{span.end_offset}".encode("utf-8")
                ).hexdigest()[:24]
            if span.span_id != expected_span_id:
                raise CandidateSourceError("candidate child span identity does not match its parent source range")
            start = span.start_offset - parent_span.start_offset
            end = span.end_offset - parent_span.start_offset
            if span.text != parent_span.text[start:end]:
                raise CandidateSourceError("candidate child text does not match its parent source range")
    return dict(sorted(
        child_map.items(),
        key=lambda item: (
            item[1].section_path,
            item[1].page_start,
            item[1].child_ordinal,
            item[0],
        ),
    ))


def _select_sources(children: tuple[ChildDocumentChunk, ...], batch_index: int) -> tuple[ChildDocumentChunk, ...]:
    start = (batch_index * _BATCH_SIZE) % len(children)
    return tuple(children[(start + offset) % len(children)] for offset in range(min(_BATCH_SIZE, len(children))))


def _prompt(remaining: int, accepted: list[EvaluationRowV2]) -> str:
    categories = Counter(row.category for row in accepted)
    outcomes = Counter(row.expected_outcome for row in accepted)
    return (
        f"Generator policy {PROMPT_VERSION}. Propose up to {_BATCH_SIZE} distinct evaluation rows for the next batch; "
        f"{remaining} rows remain toward the target. Cover categories {_CATEGORIES} and outcomes {_OUTCOMES}; "
        f"current counts are categories={dict(categories)}, outcomes={dict(outcomes)}. "
        "Return JSON with a candidates array. Each row has question, category, expected_outcome, "
        "expected_answer_or_rubric, supports, and optional scope_rationale. expected_outcome is answer, qualified, "
        "or evidence_refusal. For answer/qualified, supports must contain source_id and an exact quote from a "
        "supplied source passage. For evidence_refusal, give a corpus-scope rationale and no supports. "
        "Do not infer a refusal from missing retrieval results. Proposals remain unapproved candidates; never claim "
        "the question or refusal has been independently reviewed. Keep questions materially distinct."
    )


def _to_row(
    corpus_version_id: str,
    draft: CandidateDraft,
    sources: dict[str, tuple[ChildDocumentChunk, str]],
) -> EvaluationRowV2 | None:
    raw_spans: dict[tuple[str, int, int], RawSourceSpan] = {}
    child_ids: list[str] = []
    parent_ids: list[str] = []
    for support in draft.supports:
        source = sources.get(support.source_id)
        if source is None:
            return None
        child, sent_text = source
        if child.corpus_version_id != corpus_version_id or support.quote not in sent_text:
            return None
        if any(span.span_id is None or span.block_id is None for span in child.source_spans):
            return None
        matching = [span for span in child.source_spans if support.quote in span.text]
        if len(matching) != 1:
            return None
        span = matching[0]
        raw = RawSourceSpan(
            span_id=span.span_id or "",
            block_id=span.block_id or "",
            physical_page=span.page,
            printed_page_label=span.printed_page_label,
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            raw_text=span.text,
        )
        raw_spans[(raw.span_id, raw.start_offset, raw.end_offset)] = raw
        child_ids.append(child.chunk_id)
        parent_ids.append(child.parent_id)
    is_refusal = draft.expected_outcome == "evidence_refusal"
    if not is_refusal and (not raw_spans or not child_ids or not parent_ids):
        return None
    item_id = "generated:" + hashlib.sha256(
        f"{corpus_version_id}\x1f{_normalize_question(draft.question)}".encode()
    ).hexdigest()[:32]
    try:
        return EvaluationRowV2(
            item_id=item_id,
            origin="generated",
            question=draft.question,
            category=draft.category,
            expected_outcome=draft.expected_outcome,
            expected_answer_or_rubric=draft.expected_answer_or_rubric,
            expected_refusal=is_refusal,
            scope_rationale=draft.scope_rationale,
            source_spans=tuple(raw_spans.values()),
            relevant_child_ids=tuple(dict.fromkeys(child_ids)),
            parent_ids=tuple(dict.fromkeys(parent_ids)),
        )
    except ValidationError:
        return None


def _normalize_question(question: str) -> str:
    value = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"[^\w]+", " ", value).strip()


def _normalize_question_key(question: str) -> str:
    tokens = _normalize_question(question).split()
    content_tokens = tuple(token for token in tokens if token not in _QUESTION_FILLER)
    return " ".join(content_tokens or tokens)


def _is_near_duplicate(question: str, existing_keys: set[str]) -> bool:
    """Reject exact normalized questions and high-overlap content-word duplicates."""
    key = _normalize_question_key(question)
    if key in existing_keys:
        return True
    tokens = set(key.split())
    if len(tokens) < _NEAR_DUPLICATE_MIN_TOKENS:
        return False
    for existing_key in existing_keys:
        existing_tokens = set(existing_key.split())
        if len(existing_tokens) < _NEAR_DUPLICATE_MIN_TOKENS:
            continue
        overlap = len(tokens & existing_tokens)
        similarity = overlap / len(tokens | existing_tokens)
        if similarity >= _NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


def _usage(tokens: int, known: bool) -> int | None:
    return tokens if known else None


def _report(
    status: Literal["published", "awaiting_budget", "failed"],
    corpus_version_id: str,
    configuration_version: str,
    model: str,
    target_count: int,
    rows: tuple[EvaluationRowV2, ...],
    rejected_count: int,
    usage_tokens: int | None,
    *,
    dataset_head: DatasetHead | None = None,
    error: str | None = None,
) -> CandidateGenerationReport:
    categories = Counter(row.category for row in rows)
    outcomes = Counter(row.expected_outcome for row in rows)
    category_floor = max(1, target_count // len(_CATEGORIES))
    outcome_floor = max(1, target_count // len(_OUTCOMES))
    gaps = tuple(
        [f"category:{category}" for category in _CATEGORIES if categories[category] < category_floor]
        + [f"outcome:{outcome}" for outcome in _OUTCOMES if outcomes[outcome] < outcome_floor]
    )
    return CandidateGenerationReport(
        status=status,
        corpus_version_id=corpus_version_id,
        generator_policy_version=PROMPT_VERSION,
        generator_configuration_version=configuration_version,
        generator_model=model,
        target_count=target_count,
        generated_count=len(rows),
        rejected_count=rejected_count,
        achievable_size=len(rows),
        coverage_by_category={category: categories[category] for category in _CATEGORIES},
        coverage_by_outcome={outcome: outcomes[outcome] for outcome in _OUTCOMES},
        gaps=gaps,
        usage_tokens=usage_tokens,
        dataset_head=dataset_head,
        error=error,
    )
