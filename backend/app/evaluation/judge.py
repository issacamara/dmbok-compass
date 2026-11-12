"""Content-free, immutable storage for unapproved semantic judge proposals."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal, Protocol

from google.api_core.exceptions import AlreadyExists
from pydantic import ConfigDict, BaseModel, Field

from model.evaluation_judge import EvaluationJudgeProposal


class SemanticJudgeProposal(BaseModel):
    """An exact-output suggestion that cannot be release evidence by itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    output_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    generator_route: str | None = Field(default=None, max_length=128)
    generator_model: str | None = Field(default=None, max_length=256)
    generator_configuration_version: str | None = Field(default=None, max_length=128)
    generator_judge_overlap: bool
    judge_route: str = Field(min_length=1, max_length=128)
    judge_model: str = Field(min_length=1, max_length=256)
    judge_configuration_version: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=128)
    rubric_version: str = Field(min_length=1, max_length=128)
    output_token_cap: int = Field(ge=1, le=32000)
    claim_support: bool
    citation_support: bool
    rubric_passed: bool
    refusal_judgment: bool
    status: Literal["unapproved"] = "unapproved"
    release_authorized: Literal[False] = False


class SemanticJudgeProposalStore(Protocol):
    def get(self, proposal_id: str) -> SemanticJudgeProposal | None: ...

    def save(self, proposal: SemanticJudgeProposal) -> SemanticJudgeProposal: ...


class SemanticJudgeProposalError(ValueError):
    """Raised when an immutable proposal identity conflicts with stored data."""


class FirestoreSemanticJudgeProposalStore:
    """Store no question, answer, passage, or model rationale in the proposal."""

    collection_id = "evaluation_semantic_judge_proposals_v1"

    def __init__(self, client: Any) -> None:
        self.proposals = client.collection(self.collection_id)

    def get(self, proposal_id: str) -> SemanticJudgeProposal | None:
        snapshot = self.proposals.document(proposal_id).get()
        return SemanticJudgeProposal.model_validate(snapshot.to_dict()) if snapshot.exists else None

    def save(self, proposal: SemanticJudgeProposal) -> SemanticJudgeProposal:
        reference = self.proposals.document(proposal.proposal_id)
        payload = proposal.model_dump(mode="json")
        try:
            reference.create(payload)
        except AlreadyExists as exc:
            snapshot = reference.get()
            if not snapshot.exists or SemanticJudgeProposal.model_validate(snapshot.to_dict()) != proposal:
                raise SemanticJudgeProposalError("judge proposal identity is already bound to other evidence") from exc
            return proposal
        return proposal


def semantic_judge_proposal_id(run_id: str, item_id: str, output_digest: str, configuration_fingerprint: str) -> str:
    identity = "\0".join((run_id, item_id, output_digest, configuration_fingerprint))
    return sha256(identity.encode("utf-8")).hexdigest()


def semantic_judge_proposal(
    *,
    run_id: str,
    item_id: str,
    output_digest: str,
    judge: EvaluationJudgeProposal,
    judge_route: str,
    judge_model: str,
    judge_configuration_version: str,
    prompt_version: str,
    rubric_version: str,
    output_token_cap: int,
    generator_route: str | None,
    generator_model: str | None,
    generator_configuration_version: str | None,
    configuration_fingerprint: str,
) -> SemanticJudgeProposal:
    overlap = (
        generator_route is not None
        and generator_model is not None
        and generator_route == judge_route
        and generator_model == judge_model
    )
    return SemanticJudgeProposal(
        proposal_id=semantic_judge_proposal_id(run_id, item_id, output_digest, configuration_fingerprint),
        run_id=run_id,
        item_id=item_id,
        output_digest=output_digest,
        generator_route=generator_route,
        generator_model=generator_model,
        generator_configuration_version=generator_configuration_version,
        generator_judge_overlap=overlap,
        judge_route=judge_route,
        judge_model=judge_model,
        judge_configuration_version=judge_configuration_version,
        prompt_version=prompt_version,
        rubric_version=rubric_version,
        output_token_cap=output_token_cap,
        claim_support=judge.claim_support,
        citation_support=judge.citation_support,
        rubric_passed=judge.rubric_passed,
        refusal_judgment=judge.refusal_judgment,
    )
