"""Versioned evaluator-authored questions and human-reviewed annotations.

Evaluation data is deliberately separate from live question handling.  The
store accepts only :class:`EvaluationDataset` records, never answer requests
or production traces, so reproducible evaluation cannot become a mechanism
for retaining user interactions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from typing import Iterable, Literal, Protocol

from app.contracts import DocumentChunk, EvaluationDataset, EvaluationReportEligibility, GoldAnnotation


class EvaluationDatasetError(ValueError):
    """Base error for invalid evaluation dataset operations."""


class ReviewTransitionError(EvaluationDatasetError):
    """Raised when an annotation review transition is not allowed."""


class GoldSubsetError(EvaluationDatasetError):
    """Raised when a dataset does not have an approvable gold subset."""


class CandidateBindingError(EvaluationDatasetError):
    """Raised when remapping data is not bound to one candidate assessment."""


class CandidateChunkLookup(Protocol):
    """Resolve chunks only within the candidate corpus named by a request."""

    def get(self, *, corpus_version_id: str, chunk_id: str) -> DocumentChunk | None: ...


@dataclass(frozen=True)
class CandidateGoldBinding:
    """Immutable candidate identities required before a gold item can be reviewed."""

    corpus_version_id: str
    candidate_snapshot_sha256: str
    source_provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.corpus_version_id:
            raise CandidateBindingError("candidate corpus version is required")
        for name, value in (
            ("candidate snapshot", self.candidate_snapshot_sha256),
            ("source provenance", self.source_provenance_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise CandidateBindingError(f"{name} must be a lowercase SHA-256 digest")

    def model_dump(self) -> dict[str, str]:
        return {
            "corpus_version_id": self.corpus_version_id,
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "source_provenance_sha256": self.source_provenance_sha256,
        }


@dataclass(frozen=True)
class CandidateChunk:
    """A candidate chunk plus the provenance needed for sponsor review."""

    chunk: DocumentChunk
    section_origin: Literal["explicit", "carried"] = "explicit"

    def __post_init__(self) -> None:
        if self.section_origin not in {"explicit", "carried"}:
            raise CandidateBindingError("section origin must be explicit or carried")


@dataclass(frozen=True)
class GoldAnnotationRemap:
    """A non-final proposal that an administrator must review explicitly."""

    item_id: str
    relevant_chunk_ids: tuple[str, ...]
    remap_status: Literal["matched", "refusal", "ambiguous", "missing", "changed", "unknown"]
    page: int | None
    section: str | None
    section_origin: Literal["explicit", "carried"] | None
    review_reasons: tuple[str, ...]
    content_hash: str | None = None

    def model_dump(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "remap_status": self.remap_status,
            "review_status": "candidate",
            "page": self.page,
            "section": self.section,
            "section_origin": self.section_origin,
            "review_reasons": list(self.review_reasons),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class CandidateGoldReview:
    """Candidate-specific remap output; it deliberately has no approval state."""

    binding: CandidateGoldBinding
    remaps: tuple[GoldAnnotationRemap, ...]
    binding_version: str = "candidate-binding-v1"
    expected_refusal: bool = False
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    chunk_hashes: dict[str, str] | None = None

    def model_dump(self) -> dict[str, object]:
        return {
            "binding": self.binding.model_dump(),
            "remaps": [remap.model_dump() for remap in self.remaps],
            "binding_version": self.binding_version,
            "expected_refusal": self.expected_refusal,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "chunk_hashes": dict(self.chunk_hashes or {}),
        }

    def remap_for(self, item_id: str) -> GoldAnnotationRemap:
        try:
            return next(remap for remap in self.remaps if remap.item_id == item_id)
        except StopIteration as exc:
            raise CandidateBindingError(f"candidate review is missing annotation {item_id}") from exc


def remap_gold_annotations(
    dataset: EvaluationDataset,
    candidate_chunks: Iterable[DocumentChunk | CandidateChunk],
    binding: CandidateGoldBinding,
) -> CandidateGoldReview:
    """Propose candidate-corpus mappings without carrying approval forward.

    A supporting passage can map only when it occurs in exactly one candidate
    chunk for the bound corpus.  Every output remains a ``candidate`` for the
    existing administrator review flow; ambiguous, missing, and changed
    support have no candidate chunk IDs to approve accidentally.
    """

    chunks = tuple(_candidate_chunk(chunk) for chunk in candidate_chunks)
    if any(chunk.chunk.corpus_version_id != binding.corpus_version_id for chunk in chunks):
        raise CandidateBindingError("candidate chunks must all belong to the bound candidate corpus")

    remaps = tuple(_remap_annotation(annotation, chunks) for annotation in dataset.gold_annotations)
    return CandidateGoldReview(binding=binding, remaps=remaps)


def candidate_review_fingerprint(review: CandidateGoldReview) -> str:
    """Return a stable binding fingerprint for durable generation metadata."""

    payload = review.model_dump()
    payload.pop("reviewed_by", None)
    payload.pop("reviewed_at", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _candidate_chunk(chunk: DocumentChunk | CandidateChunk) -> CandidateChunk:
    return chunk if isinstance(chunk, CandidateChunk) else CandidateChunk(chunk=chunk)


def _remap_annotation(
    annotation: GoldAnnotation, chunks: tuple[CandidateChunk, ...]
) -> GoldAnnotationRemap:
    support = _normalise(annotation.supporting_passage or "")
    matches = tuple(chunk for chunk in chunks if support and support in _normalise(chunk.chunk.text))
    if len(matches) == 1:
        match = matches[0]
        reasons = ["verify_one_based_page_and_section"]
        if match.section_origin == "carried":
            reasons.append("review_carried_heading")
        if match.chunk.section == "Unclassified":
            reasons.append("review_unclassified_section")
        return GoldAnnotationRemap(
            item_id=annotation.item_id,
            relevant_chunk_ids=(match.chunk.chunk_id,),
            remap_status="matched",
            page=match.chunk.page,
            section=match.chunk.section,
            section_origin=match.section_origin,
            review_reasons=tuple(reasons),
        )
    if len(matches) > 1:
        return GoldAnnotationRemap(
            item_id=annotation.item_id,
            relevant_chunk_ids=(),
            remap_status="ambiguous",
            page=None,
            section=None,
            section_origin=None,
            review_reasons=("ambiguous_support",),
        )
    if not support:
        status: Literal["missing", "changed", "unknown"] = "missing"
        reason = "missing_supporting_passage"
    elif any(chunk.chunk.chunk_id in annotation.relevant_chunk_ids for chunk in chunks):
        status = "changed"
        reason = "changed_support"
    else:
        status = "unknown"
        reason = "unknown_support"
    return GoldAnnotationRemap(
        item_id=annotation.item_id,
        relevant_chunk_ids=(),
        remap_status=status,
        page=None,
        section=None,
        section_origin=None,
        review_reasons=(reason,),
    )


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


class EvaluationDatasetStore(Protocol):
    def create(self, dataset: EvaluationDataset) -> EvaluationDataset: ...

    def get(self, dataset_version_id: str) -> EvaluationDataset | None: ...

    def review_annotation(
        self, dataset_version_id: str, item_id: str, review_status: str
    ) -> EvaluationDataset: ...

    def approve_gold_subset(self, dataset_version_id: str) -> EvaluationDataset: ...


class InMemoryEvaluationDatasetStore:
    """Reference store for local development and evaluation-job tests.

    Dataset versions are immutable at creation.  The only permitted later
    change is the explicit human review transition of an existing annotation.
    """

    MIN_GOLD_ITEMS = 30
    MAX_GOLD_ITEMS = 50
    _TRANSITIONS = {
        "candidate": frozenset({"approved", "rejected"}),
        "approved": frozenset(),
        "rejected": frozenset(),
    }

    def __init__(self) -> None:
        self._datasets: dict[str, EvaluationDataset] = {}
        self._gold_approved: set[str] = set()
        self._active_dataset_version_id: str | None = None
        self._lock = RLock()

    def create(self, dataset: EvaluationDataset) -> EvaluationDataset:
        self._validate_dataset(dataset)
        with self._lock:
            existing = self._datasets.get(dataset.dataset_version_id)
            if existing is not None:
                if existing != dataset:
                    raise EvaluationDatasetError(
                        "dataset_version_id is already bound to another dataset"
                    )
                return deepcopy(existing)
            self._datasets[dataset.dataset_version_id] = dataset.model_copy(deep=True)
            self._active_dataset_version_id = dataset.dataset_version_id
            return deepcopy(dataset)

    def get(self, dataset_version_id: str) -> EvaluationDataset | None:
        with self._lock:
            dataset = self._datasets.get(dataset_version_id)
            return deepcopy(dataset) if dataset is not None else None

    def active_dataset(self) -> EvaluationDataset | None:
        with self._lock:
            if self._active_dataset_version_id is None:
                return None
            return deepcopy(self._datasets[self._active_dataset_version_id])

    def report_eligibility(self, dataset_version_id: str) -> EvaluationReportEligibility:
        with self._lock:
            dataset = self._require(dataset_version_id)
            approved_gold_count = sum(
                annotation.review_status == "approved" for annotation in dataset.gold_annotations
            )
            is_current = dataset_version_id == self._active_dataset_version_id
            return EvaluationReportEligibility(
                generation_id=dataset_version_id,
                eligibility="release_evidence" if is_current and self.MIN_GOLD_ITEMS <= approved_gold_count <= self.MAX_GOLD_ITEMS else "exploratory",
                approved_gold_count=approved_gold_count,
                is_current_generation=is_current,
                is_superseded=not is_current,
            )

    def review_annotation(
        self, dataset_version_id: str, item_id: str, review_status: str
    ) -> EvaluationDataset:
        if review_status not in {"candidate", "approved", "rejected"}:
            raise ReviewTransitionError(f"unknown review status: {review_status}")
        with self._lock:
            dataset = self._require(dataset_version_id)
            annotations = list(dataset.gold_annotations)
            try:
                index = next(i for i, annotation in enumerate(annotations) if annotation.item_id == item_id)
            except StopIteration as exc:
                raise EvaluationDatasetError(f"unknown annotation item: {item_id}") from exc

            current = annotations[index].review_status
            if review_status not in self._TRANSITIONS[current]:
                raise ReviewTransitionError(
                    f"cannot transition annotation {item_id} from {current} to {review_status}"
                )
            annotations[index] = annotations[index].model_copy(update={"review_status": review_status})
            updated = dataset.model_copy(update={"gold_annotations": annotations}, deep=True)
            self._datasets[dataset_version_id] = updated
            return deepcopy(updated)

    def approve_gold_subset(self, dataset_version_id: str) -> EvaluationDataset:
        with self._lock:
            dataset = self._require(dataset_version_id)
            approved = [
                annotation for annotation in dataset.gold_annotations if annotation.review_status == "approved"
            ]
            count = len(approved)
            if not self.MIN_GOLD_ITEMS <= count <= self.MAX_GOLD_ITEMS:
                raise GoldSubsetError("the approved gold subset must contain between 30 and 50 items")
            self._gold_approved.add(dataset_version_id)
            return deepcopy(dataset)

    def is_gold_subset_approved(self, dataset_version_id: str) -> bool:
        with self._lock:
            return dataset_version_id in self._gold_approved

    def _require(self, dataset_version_id: str) -> EvaluationDataset:
        dataset = self._datasets.get(dataset_version_id)
        if dataset is None:
            raise EvaluationDatasetError(f"unknown dataset version: {dataset_version_id}")
        return dataset

    @staticmethod
    def _validate_dataset(dataset: EvaluationDataset) -> None:
        item_ids = [item.item_id for item in dataset.items]
        if len(item_ids) != len(set(item_ids)):
            raise EvaluationDatasetError("dataset contains duplicate item IDs")
        item_versions = {item.dataset_version_id for item in dataset.items}
        if item_versions != {dataset.dataset_version_id}:
            raise EvaluationDatasetError("every item must belong to the dataset version")
        annotation_ids = [annotation.item_id for annotation in dataset.gold_annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise EvaluationDatasetError("dataset contains duplicate annotations")
        if not set(annotation_ids).issubset(item_ids):
            raise EvaluationDatasetError("every annotation must reference a dataset item")
