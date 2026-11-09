"""Versioned v2.9 domain records; provider and library objects stay internal."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .api import ContractModel, QuestionCategory

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[int, Field(strict=True, ge=1)]


class FrozenContractModel(ContractModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class ParsedBlock(FrozenContractModel):
    block_id: Identifier
    physical_page: int = Field(strict=True, ge=1)
    printed_page_label: str | None = Field(default=None, min_length=1, max_length=128)
    reading_order: int = Field(strict=True, ge=0)
    kind: Literal["text", "table", "list"]
    raw_text: str = Field(min_length=1, max_length=20000)
    table_rows: tuple[tuple[str, ...], ...] = ()
    list_items: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "ParsedBlock":
        if self.kind == "table" and (not self.table_rows or any(not row for row in self.table_rows)):
            raise ValueError("table blocks require rows")
        if self.kind == "list" and not self.list_items:
            raise ValueError("list blocks require items")
        if (self.kind != "table" and self.table_rows) or (self.kind != "list" and self.list_items):
            raise ValueError("block structure must match its kind")
        return self


class ParsingArtifact(FrozenContractModel):
    schema_version: Literal["parsing-artifact-v1"] = "parsing-artifact-v1"
    source_generation: Identifier
    source_sha256: Digest
    output_sha256: Digest
    blocks: tuple[ParsedBlock, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blocks(self) -> "ParsingArtifact":
        if len({block.block_id for block in self.blocks}) != len(self.blocks):
            raise ValueError("block IDs must be unique")
        if [block.reading_order for block in self.blocks] != list(range(len(self.blocks))):
            raise ValueError("reading order must be contiguous")
        if [block.physical_page for block in self.blocks] != sorted(block.physical_page for block in self.blocks):
            raise ValueError("blocks must follow physical page order")
        return self


class RawSourceSpan(FrozenContractModel):
    span_id: Identifier
    block_id: Identifier
    physical_page: int = Field(strict=True, ge=1)
    printed_page_label: str | None = Field(default=None, min_length=1, max_length=128)
    start_offset: int = Field(strict=True, ge=0)
    end_offset: int = Field(strict=True, gt=0)
    raw_text: str = Field(min_length=1, max_length=20000)

    @model_validator(mode="after")
    def validate_offsets(self) -> "RawSourceSpan":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class StructureReviewRevision(FrozenContractModel):
    schema_version: Literal["structure-review-v1"] = "structure-review-v1"
    attempt_id: Identifier
    revision: Revision
    parsing_output_sha256: Digest
    map_sha256: Digest
    actor_id: Identifier
    reviewed_at: AwareDatetime


class SourceTextMapping(FrozenContractModel):
    target_start: int = Field(strict=True, ge=0)
    target_end: int = Field(strict=True, gt=0)
    source_span_id: Identifier | None = None
    source_start: int | None = Field(default=None, strict=True, ge=0)
    source_end: int | None = Field(default=None, strict=True, gt=0)
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_mapping(self) -> "SourceTextMapping":
        if self.target_end <= self.target_start:
            raise ValueError("target offsets must be ordered")
        if self.synthetic:
            if self.source_span_id is not None or self.source_start is not None or self.source_end is not None:
                raise ValueError("synthetic text cannot claim source offsets")
        elif self.source_span_id is None or self.source_start is None or self.source_end is None or self.source_end <= self.source_start:
            raise ValueError("source text requires ordered source offsets")
        return self


class GenerationContextParent(FrozenContractModel):
    parent_id: Identifier
    spans: tuple[RawSourceSpan, ...] = Field(min_length=1)
    backing_child_ids: tuple[Identifier, ...] = Field(min_length=1)
    embedding_text: str = Field(min_length=1, max_length=20000)
    normalization_mappings: tuple[SourceTextMapping, ...] = ()

    @model_validator(mode="after")
    def validate_mappings(self) -> "GenerationContextParent":
        span_ids = {span.span_id for span in self.spans}
        if len(span_ids) != len(self.spans) or [span.physical_page for span in self.spans] != sorted(span.physical_page for span in self.spans):
            raise ValueError("spans must have unique IDs and follow page order")
        if any(mapping.source_span_id not in span_ids for mapping in self.normalization_mappings if not mapping.synthetic):
            raise ValueError("normalization mappings must reference supplied spans")
        if any(mapping.source_end > len(next(span.raw_text for span in self.spans if span.span_id == mapping.source_span_id)) for mapping in self.normalization_mappings if not mapping.synthetic):
            raise ValueError("normalization mapping exceeds source span")
        if any(mapping.target_end > len(self.embedding_text) for mapping in self.normalization_mappings):
            raise ValueError("normalization mapping exceeds target text")
        return self


class GenerationContext(FrozenContractModel):
    schema_version: Literal["generation-context-v1"] = "generation-context-v1"
    corpus_version_id: Identifier
    parents: tuple[GenerationContextParent, ...] = Field(min_length=1)


class EvaluationRowV2(FrozenContractModel):
    schema_version: Literal["evaluation-row-v2"] = "evaluation-row-v2"
    item_id: Identifier
    origin: Literal["generated", "admin", "imported"]
    question: str = Field(min_length=1, max_length=2000)
    category: QuestionCategory
    expected_outcome: Literal["answer", "qualified", "evidence_refusal"]
    expected_answer_or_rubric: str = Field(min_length=1, max_length=10000)
    expected_refusal: bool
    scope_rationale: str | None = Field(default=None, min_length=1, max_length=5000)
    source_spans: tuple[RawSourceSpan, ...] = Field(default_factory=tuple)
    relevant_child_ids: tuple[Identifier, ...] = Field(default_factory=tuple)
    parent_ids: tuple[Identifier, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_support(self) -> "EvaluationRowV2":
        if self.expected_refusal != (self.expected_outcome == "evidence_refusal"):
            raise ValueError("expected_refusal must match expected_outcome")
        if self.expected_refusal:
            if not self.scope_rationale or self.source_spans or self.relevant_child_ids or self.parent_ids:
                raise ValueError("refusal requires a rationale and zero support bindings")
        elif not self.source_spans or not self.relevant_child_ids or not self.parent_ids:
            raise ValueError("answerable rows require spans and child/parent bindings")
        return self


class DatasetHead(FrozenContractModel):
    schema_version: Literal["dataset-head-v1"] = "dataset-head-v1"
    corpus_version_id: Identifier
    revision: Revision
    dataset_sha256: Digest


class DatasetExclusion(FrozenContractModel):
    schema_version: Literal["dataset-exclusion-v1"] = "dataset-exclusion-v1"
    corpus_version_id: Identifier
    item_id: Identifier
    lineage_id: Identifier
    normalized_question_sha256: Digest
    revision: Revision
    actor_id: Identifier
    excluded_at: AwareDatetime


class AllowanceReservation(FrozenContractModel):
    schema_version: Literal["allowance-reservation-v1"] = "allowance-reservation-v1"
    reservation_id: Identifier
    operation_key: Identifier
    kind: Literal["parsing", "generation", "judging"]
    allowance_observed_at: AwareDatetime
    allowance_expires_at: AwareDatetime
    reserved_units: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def validate_observation(self) -> "AllowanceReservation":
        if self.allowance_expires_at <= self.allowance_observed_at:
            raise ValueError("allowance expiry must follow observation")
        return self


class CandidateDatasetBinding(FrozenContractModel):
    kind: Literal["candidate"] = "candidate"
    attempt_id: Identifier
    corpus_version_id: Identifier
    dataset_revision: Revision


class ServingDatasetBinding(FrozenContractModel):
    kind: Literal["serving"] = "serving"
    release_id: Identifier
    corpus_version_id: Identifier
    dataset_revision: Revision


class EvaluationRunSnapshot(FrozenContractModel):
    schema_version: Literal["evaluation-run-snapshot-v1"] = "evaluation-run-snapshot-v1"
    run_id: Identifier
    binding: CandidateDatasetBinding | ServingDatasetBinding = Field(discriminator="kind")
    configuration_version_id: Identifier
    selection: Literal["full", "subset"]
    items: tuple[EvaluationRowV2, ...] = Field(min_length=1)
    snapshot_sha256: Digest

    @model_validator(mode="after")
    def validate_items(self) -> "EvaluationRunSnapshot":
        if len({row.item_id for row in self.items}) != len(self.items):
            raise ValueError("snapshot item IDs must be unique")
        return self


class CorpusStagingAttemptV2(FrozenContractModel):
    schema_version: Literal["corpus-staging-attempt-v2"] = "corpus-staging-attempt-v2"
    attempt_id: Identifier
    actor_id: Identifier
    source_generation: Identifier
    source_sha256: Digest
    parser_route: Identifier
    parser_settings_sha256: Digest
    remote_job_id: Identifier | None = None
    parsing_artifact_id: Identifier | None = None
    parsing_output_sha256: Digest | None = None
    review_revision: Revision | None = None
    approval_artifact_id: Identifier | None = None
    candidate_corpus_version_id: Identifier | None = None
    status: Literal["validating", "awaiting_budget", "parsing", "awaiting_review", "queued", "running", "staged", "failed"]

    @model_validator(mode="after")
    def validate_artifact(self) -> "CorpusStagingAttemptV2":
        if (self.parsing_artifact_id is None) != (self.parsing_output_sha256 is None):
            raise ValueError("parsing artifact and digest must be paired")
        if self.review_revision is not None and self.parsing_artifact_id is None:
            raise ValueError("review revision requires a parsing artifact")
        if self.approval_artifact_id is not None and self.review_revision is None:
            raise ValueError("approval requires a review revision")
        return self


class DatasetRevisionConflict(FrozenContractModel):
    code: Literal["dataset_revision_conflict"] = "dataset_revision_conflict"
    expected_revision: Revision
    current_revision: Revision
    message: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_conflict(self) -> "DatasetRevisionConflict":
        if self.expected_revision == self.current_revision:
            raise ValueError("conflict requires different revisions")
        return self
